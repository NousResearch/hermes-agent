"""Fill zero-valued bake misses for review, preserving original atlases."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
ROOT=Path(__file__).resolve().parents[1]/'public/lunar-city/multiview-2026-09-07'
for folder in ROOT.iterdir():
 if not folder.is_dir():continue
 prefix='owl-' if folder.name=='owl-librarian' else ''
 paths={'basecolor':folder/('owl-basecolor-detail.png' if prefix else 'basecolor-review.png'),
        'normal':folder/('owl-normal-detail.png' if prefix else 'normal-review.png')}
 if not all(p.exists() for p in paths.values()):continue
 receipt={}
 for kind,path in paths.items():
  a=np.array(Image.open(path).convert('RGB'));missing=np.max(a,axis=2)==0
  if missing.all():raise ValueError(f'{path}: completely empty atlas')
  nearest=distance_transform_edt(missing,return_distances=False,return_indices=True)
  a[missing]=a[nearest[0,missing],nearest[1,missing]]
  output=folder/f'{kind}-filled-review.png';Image.fromarray(a).save(output)
  receipt[kind]={'input':path.name,'output':output.name,'zeroPixelsFilledIncludingOutsideUvIslands':int(missing.sum())}
 (folder/'bake-miss-fill.json').write_text(json.dumps({'method':'nearest valid texel for exactly zero pixels; includes unused atlas background',
 'status':'review repair; large projection errors still require artist cleanup','textures':receipt},indent=2)+'\n')
 print(folder.name,receipt,flush=True)
