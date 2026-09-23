from pathlib import Path
import json,subprocess,sys,time
SCRIPTS=Path(__file__).resolve().parent;ROOT=SCRIPTS.parent/'public/lunar-city/worker-multiview-2026-09-07'
rows=json.loads((ROOT/'worker-intake-manifest.json').read_text());pending={r['id'] for r in rows}
while pending:
 for asset in list(pending):
  folder=ROOT/asset
  if (folder/'color-study-v4.json').exists():pending.remove(asset);continue
  if not (folder/'generation.json').exists():continue
  with (folder/'projection.log').open('w') as log:
   subprocess.run([sys.executable,str(SCRIPTS/'lunar_four_view_colors.py'),asset,str(folder)],stdout=log,stderr=subprocess.STDOUT,check=True)
  pending.remove(asset);print('Projected',asset,flush=True)
 if pending:time.sleep(5)
