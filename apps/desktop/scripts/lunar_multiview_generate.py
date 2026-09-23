"""Generate card-derived geometry with explicit provenance and metric dimensions.

Use the isolated .lunar-runtime environment. Generated meshes require visual
review, material work and animation before they can become production assets.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[3]
RUNTIME = ROOT / '.lunar-runtime'
OUTPUT = ROOT / 'apps/desktop/public/lunar-city/multiview-2026-09-07'
SOURCE = Path('/Users/mikedemott/.codex/generated_images/01a069cb-5f95-7040-a5d1-5832731e13f4')
LEADERS = {
    'owl-librarian': ('exec-6a3122cb-2538-4777-9d5e-75002988b2e5.png', 1.5),
    'elephant-memory': ('exec-239f67bb-f45e-46b5-8f85-a99f0d5f12fb.png', 2.3),
    'cat-arts': ('exec-f0b3b64d-732e-4f07-a7a7-e9e2af970fe7.png', 1.6),
    'fox-scientist': ('exec-77739e6c-1029-44c4-b241-8362bcc52bd2.png', 1.75),
    'capybara-revenue': ('exec-f1e0499f-dfb6-4449-88df-9357d9a180b9.png', 1.7),
    'lion-steward': ('exec-3a25ab22-6ed2-4170-943f-11133401149d.png', 2.0),
    'beaver-architect': ('exec-cf368b49-01d0-4f3a-a9a6-db15556622f2.png', 1.6),
    'monkey-poet': ('exec-efa2fe66-b4d7-411a-931e-2b428406c0db.png', 1.65),
}

def prepare(asset):
    from PIL import Image
    import numpy as np
    from rembg import new_session, remove
    os.environ['U2NET_HOME'] = str(RUNTIME / 'background-models')
    source_name, height = LEADERS[asset]
    source = SOURCE / source_name
    folder = OUTPUT / asset
    folder.mkdir(parents=True, exist_ok=True)
    source_bytes = source.read_bytes()
    (folder / 'source-card.png').write_bytes(source_bytes)
    image = Image.open(source).convert('RGB')
    w,h = image.size
    # The drawn divider is not always exactly at half height. Overlap the
    # lower crop slightly; remove detached label components after segmentation.
    boxes = {'top': (0,0,w//2,int(h*.46)), 'left': (w//2,0,w,int(h*.46)),
             'front': (0,h//2-16,w//2,int(h*.95)), 'back': (w//2,h//2-16,w,int(h*.95))}
    session = new_session('u2net')
    receipt = {'asset':asset, 'sourceSha256':hashlib.sha256(source_bytes).hexdigest(),
               'sourceOriginal':str(source), 'targetHeightMetres':height,
               'dimensionBasis':'chosen stylized character height; card has no measured dimensions',
               'topViewUse':'QA only; unsupported as a Hunyuan elevation', 'views':{},
               'status':'prepared_requires_mask_and_orientation_review'}
    for name, box in boxes.items():
        cleaned = remove(image.crop(box),session=session)
        alpha = np.array(cleaned.getchannel('A'))
        from scipy.ndimage import label, find_objects
        components, count = label(alpha > 127)
        sizes = np.bincount(components.ravel()); sizes[0] = 0
        if count == 0: raise ValueError(f'{name} has no foreground')
        main = find_objects(components)[int(sizes.argmax())-1]
        keep = np.zeros_like(alpha)
        keep[main] = alpha[main]
        cleaned.putalpha(Image.fromarray(keep))
        alpha = keep
        coverage = float((alpha>127).mean())
        if not .04 < coverage < .90:
            raise ValueError(f'{name} suspicious foreground coverage: {coverage}')
        cleaned.save(folder / f'{name}.png')
        receipt['views'][name]={'crop':box,'foregroundCoverage':coverage,
            'sha256':hashlib.sha256((folder/f'{name}.png').read_bytes()).hexdigest()}
    (folder/'intake.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)

def generate(asset):
    import torch
    from PIL import Image
    import numpy as np
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    if not torch.backends.mps.is_available():
        raise RuntimeError('Apple Metal unavailable. No silent CPU or quality fallback.')
    folder=OUTPUT/asset
    receipt=json.loads((folder/'intake.json').read_text())
    model=RUNTIME/'models/Hunyuan3D-2mv/hunyuan3d-dit-v2-mv-turbo'
    print('Loading multi-view model on Apple Metal',flush=True)
    pipe=Hunyuan3DDiTFlowMatchingPipeline.from_single_file(
        str(model/'model.fp16.safetensors'),str(model/'config.yaml'),
        device='mps',dtype=torch.float16,use_safetensors=True)
    images={name:Image.open(folder/f'{name}.png').convert('RGBA') for name in ('front','left','back')}
    started=time.time()
    mesh=pipe(image=images,num_inference_steps=5,octree_resolution=256,
              num_chunks=10000,generator=torch.Generator(device='cpu').manual_seed(12345),
              output_type='trimesh')[0]
    if mesh is None or not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
        raise RuntimeError('Invalid generated mesh')
    # Hunyuan output is Y-up. GLB metres are checked again on Blender import.
    low,high=mesh.bounds
    if high[1]-low[1] <= 0: raise RuntimeError('Invalid vertical extent')
    mesh.apply_translation([-float((low[0]+high[0])/2),-float(low[1]),-float((low[2]+high[2])/2)])
    mesh.apply_scale(receipt['targetHeightMetres']/float(high[1]-low[1]))
    mesh.export(folder/'shape-metres.glb')
    receipt.update(status='shape_generated_requires_visual_review_textures_rig',
                   modelRevision='3a761b539b29fe4ff64714813aa9560fd66f5de0',
                   codeRevision='f8db63096c8282cb27354314d896feba5ba6ff8a',
                   steps=5,octreeResolution=256,seed=12345,device='mps',
                   elapsedSeconds=time.time()-started,faces=len(mesh.faces),
                   boundsMetres=mesh.bounds.tolist(),upAxis='Y',
                   outputSha256=hashlib.sha256((folder/'shape-metres.glb').read_bytes()).hexdigest())
    (folder/'generation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','generate'])
    parser.add_argument('asset',choices=LEADERS)
    args=parser.parse_args()
    (prepare if args.stage=='prepare' else generate)(args.asset)
