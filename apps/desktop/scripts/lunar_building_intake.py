"""Recover existing multiview building meshes and their exact turnaround cards."""
from pathlib import Path
import runpy,json,hashlib
from PIL import Image
import trimesh
BASE=Path(__file__).resolve().parents[3];ROOT=BASE/'apps/desktop/public/lunar-city/building-multiview-2026-09-07'
SOURCE=Path('/Users/mikedemott/.codex/generated_images/01a069cb-5f95-7040-a5d1-5832731e13f4')
OLD=BASE/'apps/desktop/public/lunar-city/generated-3d/hunyuan2mv-reference-2026-09-04'
rows=[('owl',16,'f0b2bb19-711c-4a70-aa9c-b3887d24890c'),('elephant',24,'fa44a25a-f6e3-4be9-ae34-a5e0bd80dff9'),('cat',10,'7c0eae54-1768-455c-81d5-538d785e7136'),('fox',12,'e639d730-b280-484e-b057-d83d7f10e422'),('capybara',16,'a356f883-9988-4840-84d8-a26b5ceb2abc'),('lion',26,'65e67b81-fae8-47e5-a39c-6da24eab789d'),('beaver',14,'9f76f574-bde3-470b-9b5b-0523439ccd51'),('monkey',12,'cb81adb2-368f-406c-91c7-a293f86ee958')]
module=runpy.run_path('/Users/mikedemott/.codex/skills/card-to-3d/scripts/multiview.py')
for asset,height,source in rows:
 folder=ROOT/asset
 if (folder/'generation.json').exists():continue
 path=SOURCE/('exec-'+source+'.png');w,h=Image.open(path).size
 crops={'top':[0,0,w//2,int(h*.46)],'left':[w//2,0,w,int(h*.46)],'front':[0,h//2-16,w//2,int(h*.95)],'back':[w//2,h//2-16,w,int(h*.95)]}
 module['prepare'](path,folder,BASE/'.lunar-runtime',height,crops)
 mesh=trimesh.load(OLD/(asset+'.glb'),force='mesh');lo,hi=mesh.bounds;mesh.apply_translation([-(lo[0]+hi[0])/2,-lo[1],-(lo[2]+hi[2])/2]);mesh.apply_scale(height/(hi[1]-lo[1]));mesh.export(folder/'shape-metres.glb')
 intake=json.loads((folder/'intake.json').read_text())
 intake.update(status='recovered_shape_requires_card_alignment_review',sourceMesh=str(OLD/(asset+'.glb')),sourceMeshSha256=hashlib.sha256((OLD/(asset+'.glb')).read_bytes()).hexdigest(),scaleBasis='Design height chosen from approximate 3m entrance proportions in turnaround; not a surveyed dimension',boundsMetres=mesh.bounds.tolist(),upAxis='Y',frontAxis='+Z',faces=len(mesh.faces),outputSha256=hashlib.sha256((folder/'shape-metres.glb').read_bytes()).hexdigest())
 (folder/'generation.json').write_text(json.dumps(intake,indent=2)+'\n');print(asset,mesh.extents,flush=True)
