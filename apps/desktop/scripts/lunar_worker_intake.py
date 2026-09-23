"""Prepare the recovered specialist worker cards with explicit target dimensions."""
from pathlib import Path
import json,runpy,contextlib
from PIL import Image
ROOT=Path(__file__).resolve().parents[3]
SOURCE=Path('/Users/mikedemott/.codex/generated_images/01a069cb-5f95-7040-a5d1-5832731e13f4')
OUTPUT=ROOT/'apps/desktop/public/lunar-city/worker-multiview-2026-09-07'
IDS={
'baseline':'eee77831-920f-4ad2-9fd0-d3eb29b7fb36',
'acceptance-release':'3d17c351-a686-4136-aadb-c3381b5813db',
'archive-acquisition':'60bd40f7-6f1a-4a34-b7f2-ab5f1975727a',
'arts-studio':'fb4ea021-27c6-4cd4-9fe1-7188d5a52d19',
'ci-repair-triage':'c332e45e-33b3-466a-9d92-24005705f32b',
'content-studio':'953a6d16-605f-42ac-a519-b14c0ba227a1',
'community-intake':'4d44d5fb-88c9-455e-aa95-fb6b619081a9',
'control-plane-incidents':'9b894a71-88e9-434c-9c42-7853dede7790',
'core-runtime-ux-repairs':'a2055f7e-b67e-4d9d-854d-169f409acf3c',
'data-performance-repairs':'fe39261f-90b8-428c-a8ff-f175b4a96a86',
'engineering-guild':'7b518a16-ba24-4731-9aad-562c3c44d48d',
'editorial-desk':'0d9ffdcb-6865-492e-8844-271aacb175dd',
'federation-council':'dfc92b3d-771f-465d-8ba5-eb793db3d97c',
'knowledge-commons':'50f7df28-babb-463d-b9df-8ae3b91a925b',
'memory-stewardship':'c4bdcbb2-dbc4-4a0a-88bd-32a0ab1a82d4',
'operations-release':'8ba821d8-6464-4f1d-b7b0-1929db026995',
'pr-merge-train':'976fa893-c70e-493f-8bcf-82e84e3d93d9',
'research-lab':'c544fd1a-7895-4199-bb4d-e57e4fbdaf61',
'research-review-board':'9028b632-d161-49dc-9ca7-61a5582927e8',
'upstream-hermes-maintenance':'8611cf22-d534-48d7-8d3c-865caca48738'}
if __name__=='__main__':
 OUTPUT.mkdir(exist_ok=True)
 module=runpy.run_path('/Users/mikedemott/.codex/skills/card-to-3d/scripts/multiview.py')
 manifest=[]
 for name,key in IDS.items():
  source=SOURCE/f'exec-{key}.png';folder=OUTPUT/name;height=.95 if name=='archive-acquisition' else 1.2
  w,h=Image.open(source).size
  boxes={'top':[0,0,w//2,int(h*.46)],'left':[w//2,0,w,int(h*.46)],'front':[0,h//2-16,w//2,int(h*.95)],'back':[w//2,h//2-16,w,int(h*.95)]}
  if not (folder/'intake.json').exists():
   with open(OUTPUT/f'{name}-intake.log','w', encoding='utf-8') as log,contextlib.redirect_stdout(log):module['prepare'](source,folder,ROOT/'.lunar-runtime',height,boxes)
  manifest.append({'id':name,'source':str(source),'heightMetres':height,'status':'mask_review_pending'})
  print('Prepared',name,flush=True)
 (OUTPUT/'worker-intake-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
