"""Finite generation queue for the twenty reviewed worker cards."""
from pathlib import Path
import json,subprocess,sys,time
BASE=Path(__file__).resolve().parents[3]
ROOT=BASE/'apps/desktop/public/lunar-city/worker-multiview-2026-09-07'
RUNTIME=BASE/'.lunar-runtime'
SCRIPT=Path('/Users/mikedemott/.codex/skills/card-to-3d/scripts/multiview.py')
rows=json.loads((ROOT/'worker-intake-manifest.json').read_text())
for row in rows:
 folder=ROOT/row['id']
 if (folder/'generation.json').exists():continue
 (ROOT/'generation-batch-status.json').write_text(json.dumps({'current':row['id'],'stage':'generating','started':time.time()})+'\n')
 with (folder/'generation.log').open('w') as log:
  result=subprocess.run([sys.executable,str(SCRIPT),'generate','--runtime',str(RUNTIME),'--output',str(folder)],stdout=log,stderr=subprocess.STDOUT)
 if result.returncode:
  (ROOT/'generation-batch-status.json').write_text(json.dumps({'current':row['id'],'stage':'failed','returncode':result.returncode})+'\n')
  raise SystemExit(result.returncode)
 print('Generated',row['id'],flush=True)
(ROOT/'generation-batch-status.json').write_text(json.dumps({'stage':'complete','assets':len(rows)})+'\n')
