"""Finite reconstruction check after the existing worker GPU batch completes."""
from pathlib import Path
import json,subprocess,sys,time
BASE=Path(__file__).resolve().parents[3]
CITY=BASE/'apps/desktop/public/lunar-city'
STATUS=CITY/'worker-multiview-2026-09-07/generation-batch-status.json'
FOLDER=CITY/'building-multiview-2026-09-07/cat/reconstruction-v2'
while True:
 state=json.loads(STATUS.read_text())
 if state['stage']=='complete':break
 if state['stage']=='failed':raise RuntimeError('Worker generation failed; inspect before continuing GPU work')
 time.sleep(15)
if not (FOLDER/'generation.json').exists():
 with (FOLDER/'generation.log').open('w') as log:
  subprocess.run([sys.executable,'/Users/mikedemott/.codex/skills/card-to-3d/scripts/multiview.py','generate','--runtime',str(BASE/'.lunar-runtime'),'--output',str(FOLDER)],stdout=log,stderr=subprocess.STDOUT,check=True)
print('Cat building reconstruction ready for visual comparison',flush=True)
