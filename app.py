import os
import time
import uuid
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
from torch import nn
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
MAX_LR_EDGE = int(os.environ.get('MAX_LR_EDGE', '384'))
if not 32 <= MAX_LR_EDGE <= 768:
    raise ValueError('MAX_LR_EDGE must be between 32 and 768.')
OUT = Path('/tmp/prism_sr')
OUT.mkdir(parents=True, exist_ok=True)

class HybridAttention(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.channel=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(c,max(c//8,4),1),nn.GELU(),nn.Conv2d(max(c//8,4),c,1),nn.Sigmoid())
        self.spatial=nn.Conv2d(2,1,7,padding=3)
    def forward(self,x):
        x=x*self.channel(x)
        return x*torch.sigmoid(self.spatial(torch.cat([x.mean(1,keepdim=True),x.amax(1,keepdim=True)],1)))

class WindowTransformer(nn.Module):
    def __init__(self,c,heads,w,offset):
        super().__init__(); self.w=w; self.offset=offset; self.heads=heads
        self.n1=nn.LayerNorm(c); self.n2=nn.LayerNorm(c)
        self.qkv=nn.Linear(c,c*3); self.proj=nn.Linear(c,c)
        self.ff=nn.Sequential(nn.Linear(c,c*2),nn.GELU(),nn.Linear(c*2,c))
        coords=torch.stack(torch.meshgrid(torch.arange(w),torch.arange(w),indexing='ij')).flatten(1)
        rel=coords[:,:,None]-coords[:,None,:]+w-1
        self.register_buffer('index',rel[0]*(2*w-1)+rel[1])
        self.bias=nn.Parameter(torch.zeros((2*w-1)**2,heads))
    def forward(self,x):
        b,c,h,k=x.shape; w=self.w; o=self.offset
        ph=(-(h+o))%w; pw=(-(k+o))%w
        z=F.pad(x,(o,pw,o,ph),mode='replicate'); hp,kp=z.shape[-2:]
        z=z.permute(0,2,3,1).reshape(b,hp//w,w,kp//w,w,c).permute(0,1,3,2,4,5).reshape(-1,w*w,c)
        qkv=self.qkv(self.n1(z)).reshape(-1,w*w,3,self.heads,c//self.heads).permute(2,0,3,1,4)
        q,key,v=qkv.unbind(0)
        bias=self.bias[self.index].permute(2,0,1).unsqueeze(0)
        a=F.scaled_dot_product_attention(q,key,v,attn_mask=bias)
        z=z+self.proj(a.transpose(1,2).reshape(-1,w*w,c))
        z=z+self.ff(self.n2(z))
        z=z.reshape(b,hp//w,kp//w,w,w,c).permute(0,1,3,2,4,5).reshape(b,hp,kp,c).permute(0,3,1,2)
        return z[:,:,o:o+h,o:o+k]

class FusionBlock(nn.Module):
    def __init__(self,c,heads,window,offset,multiscale,hybrid,transformer):
        super().__init__()
        specs=[(3,1),(5,1),(3,2)] if multiscale else [(3,1)]
        self.branches=nn.ModuleList([nn.Sequential(nn.Conv2d(c,c,k,padding=(k//2)*d,dilation=d),nn.GELU(),HybridAttention(c) if hybrid else nn.Identity()) for k,d in specs])
        self.fuse=nn.Conv2d(c*len(specs),c,1)
        self.trans=WindowTransformer(c,heads,window,offset) if transformer else nn.Identity()
    def forward(self,x):
        z=self.fuse(torch.cat([branch(x) for branch in self.branches],1))
        return x+.1*self.trans(z)

class PrismSR(nn.Module):
    def __init__(self,scale=2,channels=48,depth=4,heads=4,window=8,multiscale=True,hybrid=True,dense=True,transformer=True):
        super().__init__(); self.scale=scale; self.dense=dense
        assert channels%heads==0 and scale in [2,3,4]
        self.stem=nn.Conv2d(3,channels,3,padding=1)
        self.blocks=nn.ModuleList([FusionBlock(channels,heads,window,(i%2)*(window//2),multiscale,hybrid,transformer) for i in range(depth)])
        self.fuse=nn.Conv2d(channels*(depth+1) if dense else channels,channels,1)
        self.up=nn.Sequential(nn.Conv2d(channels,channels*scale**2,3,padding=1),nn.PixelShuffle(scale),nn.GELU(),nn.Conv2d(channels,3,3,padding=1))
        nn.init.zeros_(self.up[-1].weight); nn.init.zeros_(self.up[-1].bias)
    def forward(self,x):
        z=self.stem(x); features=[z]
        for block in self.blocks: z=block(z); features.append(z)
        z=self.fuse(torch.cat(features,1) if self.dense else z)
        return F.interpolate(x,scale_factor=self.scale,mode='bicubic',align_corners=False)+self.up(z)

def tensor(im):
    return torch.from_numpy(np.asarray(im,dtype=np.float32).copy()/255.).permute(2,0,1)

def pil(t):
    return Image.fromarray((t.detach().cpu().clamp(0,1).permute(1,2,0).numpy()*255).round().astype('uint8'))

def metrics(pred,target,shave):
    a=pred.detach().cpu().clamp(0,1).permute(1,2,0).numpy()
    b=target.detach().cpu().clamp(0,1).permute(1,2,0).numpy()
    if shave: a,b=a[shave:-shave,shave:-shave],b[shave:-shave,shave:-shave]
    return peak_signal_noise_ratio(b,a,data_range=1),structural_similarity(b,a,data_range=1,channel_axis=2)

CHECKPOINT = Path(__file__).resolve().parent / 'best.pt'
if not CHECKPOINT.is_file():
    raise FileNotFoundError('Upload your Kaggle best.pt beside app.py in the Space Files tab, then restart.')
# Restricted checkpoint loading: best.pt from this notebook uses tensors and plain metadata.
ck = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
if not isinstance(ck, dict) or not {'model', 'config', 'epoch'} <= ck.keys():
    raise ValueError('Expected the best.pt checkpoint produced by PRISM_SR_Kaggle.ipynb.')
CFG = ck['config']
model = PrismSR(**CFG)
model.load_state_dict(ck['model'], strict=True)
model.to(DEVICE).eval()
print(f'Loaded checkpoint epoch {ck["epoch"]} on {DEVICE}; scale x{CFG["scale"]}')

@torch.inference_mode()
def tiled(low,tile=128,halo=24):
    s=CFG['scale']; _,_,h,w=low.shape
    out=torch.zeros(1,3,h*s,w*s,device='cpu')
    for y in range(0,h,tile):
        for x in range(0,w,tile):
            y1,x1=max(0,y-halo),max(0,x-halo)
            y2,x2=min(h,y+tile+halo),min(w,x+tile+halo)
            pred=model(low[:,:,y1:y2,x1:x2].to(DEVICE)).cpu()
            hh,ww=min(tile,h-y),min(tile,w-x)
            out[:,:,y*s:(y+hh)*s,x*s:(x+ww)*s]=pred[:,:,(y-y1)*s:(y-y1+hh)*s,(x-x1)*s:(x-x1+ww)*s]
    return out.clamp(0,1)

import gradio as gr
EXPORT=OUT/'exports'; EXPORT.mkdir(exist_ok=True)
def enhance(upload,mode):
    if upload is None: raise gr.Error('Upload an image first.')
    # Remove exports older than one hour, before making this output.
    for old in EXPORT.glob('sr_*.png'):
        if time.time()-old.stat().st_mtime > 3600: old.unlink(missing_ok=True)
    im=ImageOps.exif_transpose(upload).convert('RGB'); original_size=im.size; s=CFG['scale']; target=None
    reference=mode=='Reference experiment'
    limit=MAX_LR_EDGE*s if reference else MAX_LR_EDGE
    if max(im.size)>limit:
        im.thumbnail((limit,limit),Image.Resampling.LANCZOS)
    if reference:
        w=im.width//s*s; h=im.height//s*s
        if min(w,h)<s*8: raise gr.Error('Use a larger image (at least 8 LR pixels per side).')
        target=im.crop((0,0,w,h)); low=target.resize((w//s,h//s),Image.Resampling.BICUBIC)
    else: low=im
    if min(low.size)<8: raise gr.Error('Use an image at least 8 pixels wide and high.')
    t=tensor(low).unsqueeze(0)
    if DEVICE.type=='cuda': torch.cuda.synchronize()
    begun=time.perf_counter(); result=tiled(t)
    if DEVICE.type=='cuda': torch.cuda.synchronize()
    elapsed=time.perf_counter()-begun
    base=F.interpolate(t,scale_factor=s,mode='bicubic',align_corners=False).clamp(0,1)
    sr=pil(result[0]); bic=pil(base[0])
    rows=[]
    if target is not None:
        for name,value in [('Bicubic',base[0]),('Proposed model',result[0])]:
            p,ss=metrics(value,tensor(target),s); rows.append([name,round(p,3),round(ss,4)])
    w,h=sr.size; size=min(192,w,h); x=(w-size)//2; y=(h-size)//2
    box=(x,y,x+size,y+size)
    crops=[(bic.crop(box),'Bicubic detail'),(sr.crop(box),'Model detail')]
    if target is not None: crops.append((target.crop(box),'Reference detail'))
    path=EXPORT/f'sr_{uuid.uuid4().hex[:10]}.png'; sr.save(path)
    status=f'**×{s} reconstruction** · {low.width}×{low.height} → {sr.width}×{sr.height} · {elapsed:.2f}s tiled inference · {DEVICE.type.upper()} · checkpoint epoch {ck["epoch"]}'
    if im.size != original_size: status+=f'\n\nInput resized from {original_size[0]}×{original_size[1]} for this hosted demo (LR edge limit: {MAX_LR_EDGE}px).'
    if target is None: status+='\n\nPSNR/SSIM require an aligned HR reference and are unavailable for this upload.'
    return (bic,sr),crops,pd.DataFrame(rows,columns=['Method','PSNR (dB)','SSIM']),status,str(path)

css="""
.gradio-container {max-width:1200px!important; margin:auto!important;}
#hero {background:linear-gradient(120deg,#102037,#123f4b);padding:32px;border-radius:20px;color:white;margin-bottom:20px;}
#hero h1,#hero p {color:white!important;}
#run {background:#137e82!important;border:0!important;color:white!important;}
"""
with gr.Blocks(theme=gr.themes.Soft(primary_hue='teal',neutral_hue='slate'),css=css,title='PRISM SR | Evaluation Studio') as demo:
    gr.HTML('<div id="hero"><p>COMPUTER VISION · SINGLE-IMAGE SUPER-RESOLUTION</p><h1>PRISM SR</h1><p>Explore detail. Compare reconstruction. Measure the difference.</p></div>')
    with gr.Tab('Reconstruction studio'):
        with gr.Row():
            with gr.Column(scale=1):
                upload=gr.Image(type='pil',label='Source image',height=300)
                mode=gr.Radio(['Enhance LR image','Reference experiment'],value='Enhance LR image',label='Experiment mode')
                gr.Markdown('Reference experiment: upload an **HR image**. We generate its LR input for a fair paired comparison.')
                run=gr.Button('Reconstruct image →',variant='primary',elem_id='run')
            with gr.Column(scale=2):
                comparison=gr.ImageSlider(label='Bicubic ← slide → Proposed model',type='pil')
                status=gr.Markdown('Load an image to begin.')
        crops=gr.Gallery(label='Matched detail crops',columns=3,height=260)
        table=gr.Dataframe(headers=['Method','PSNR (dB)','SSIM'],interactive=False,label='Reference metrics · RGB · border-shaved')
        download=gr.File(label='Download reconstructed PNG')
    with gr.Tab('Architecture & evidence'):
        gr.Markdown('### From pixels to reconstruction\n3×3 stem → multi-scale branches → channel/spatial attention → dense fusion → window Transformer → PixelShuffle + bicubic residual.\n\n**Checkpoint:** best.pt contains trained weights and configuration. Training logs remain in the Kaggle project. Report measured improvements; attention and reconstruction do not guarantee recovery of the original lost detail.')
        gr.JSON(CFG,label='Model configuration')
        gr.Markdown('**Protocol:** PIL bicubic LR generation; PyTorch bicubic baseline; RGB [0,1]; scale-pixel border shave. Validation uses center crops. UI uses tiled inference. Do not mix these scores with Y-channel full-image benchmark tables.')
    run.click(enhance,[upload,mode],[comparison,crops,table,status,download],concurrency_limit=1)
if __name__ == '__main__':
    demo.queue(max_size=8).launch(show_error=True, max_file_size='20mb')
