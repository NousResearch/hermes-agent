"""Finish reviewed card materials while specialist rig fitting remains separate."""
import bpy,runpy,json,traceback
from pathlib import Path
SCRIPTS=Path(__file__).resolve().parent
ROOT=SCRIPTS.parent/'public/lunar-city/worker-multiview-2026-09-07'
ASSETS=[r['id'] for r in json.loads((ROOT/'worker-intake-manifest.json').read_text())]
def tick():
 if bpy.context.mode!='OBJECT':return 5.0
 pending=[a for a in ASSETS if not (ROOT/a/'material-review.json').exists()]
 if not pending:
  (ROOT/'material-batch-status.json').write_text(json.dumps({'stage':'complete_requires_visual_review'})+'\n');return None
 for asset in pending:
  if not (ROOT/asset/'color-study-v4.json').exists():continue
  try:
   (ROOT/'material-batch-status.json').write_text(json.dumps({'asset':asset,'stage':'baking'})+'\n')
   runpy.run_path(str(SCRIPTS/'lunar_bake_asset.py'),init_globals={'ASSET':asset,'ASSET_ROOT':str(ROOT),'FACE_BUDGET':30000})
   (ROOT/'material-batch-status.json').write_text(json.dumps({'asset':asset,'stage':'saved_requires_visual_review'})+'\n')
  except Exception:
   (ROOT/'material-batch-error.txt').write_text(traceback.format_exc());raise
  return 30.0
 return 10.0
previous=bpy.app.driver_namespace.get('lunar_worker_materials')
if previous and bpy.app.timers.is_registered(previous):raise RuntimeError('Worker material queue already running')
bpy.app.timers.register(tick,first_interval=2.0);bpy.app.driver_namespace['lunar_worker_materials']=tick
