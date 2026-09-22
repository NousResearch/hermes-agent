import bpy,runpy,json,traceback
from pathlib import Path
SCRIPTS=Path(__file__).resolve().parent;ROOT=SCRIPTS.parent/'public/lunar-city/worker-multiview-2026-09-07'
ASSETS=[r['id'] for r in json.loads((ROOT/'worker-intake-manifest.json').read_text())]
def tick():
 if bpy.context.mode!='OBJECT':return 5.0
 pending=[a for a in ASSETS if not (ROOT/a/'rig-export-validation.json').exists()]
 if not pending:return None
 for asset in pending:
  if not (ROOT/asset/'color-study-v4.json').exists():continue
  try:
   for script,stage in [('lunar_bake_asset.py','uv_bake'),('lunar_worker_rig.py','rig'),('lunar_surface_finish.py','surface_finish'),('lunar_worker_animation_export.py','animation_export')]:
    (ROOT/'finish-batch-status.json').write_text(json.dumps({'asset':asset,'stage':stage})+'\n')
    runpy.run_path(str(SCRIPTS/script),init_globals={'ASSET':asset,'ASSET_ROOT':str(ROOT),'FACE_BUDGET':30000})
   (ROOT/'finish-batch-status.json').write_text(json.dumps({'asset':asset,'stage':'exported_requires_visual_review'})+'\n')
  except Exception:
   (ROOT/'finish-batch-error.txt').write_text(traceback.format_exc());raise
  return 5.0
 return 5.0
previous=bpy.app.driver_namespace.get('lunar_worker_finish')
if previous and bpy.app.timers.is_registered(previous):
 raise RuntimeError('Worker finish queue is already running')
bpy.app.timers.register(tick,first_interval=2.0);bpy.app.driver_namespace['lunar_worker_finish']=tick
