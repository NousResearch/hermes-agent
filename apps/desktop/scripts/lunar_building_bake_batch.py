import bpy,runpy,json,traceback
from pathlib import Path
SCRIPTS=Path(__file__).resolve().parent
ROOT=SCRIPTS.parent/'public/lunar-city/building-multiview-2026-09-07'
queue=iter(a for a in ['owl','elephant','cat','fox','capybara','lion','beaver','monkey'] if not (ROOT/a/'building-finish-review.json').exists())
def tick():
 try:asset=next(queue)
 except StopIteration:return None
 try:
  (ROOT/'bake-batch-status.json').write_text(json.dumps({'asset':asset,'stage':'baking'})+'\n')
  if not (ROOT/asset/'material-review.json').exists():
   runpy.run_path(str(SCRIPTS/'lunar_bake_asset.py'),init_globals={'ASSET':asset,'ASSET_ROOT':str(ROOT),'FACE_BUDGET':80000})
  runpy.run_path(str(SCRIPTS/'lunar_building_surface_finish.py'),init_globals={'ASSET':asset})
  (ROOT/'bake-batch-status.json').write_text(json.dumps({'asset':asset,'stage':'exported_requires_visual_review'})+'\n')
 except Exception:
  (ROOT/'bake-batch-error.txt').write_text(traceback.format_exc());raise
 return 30.0
previous=bpy.app.driver_namespace.get('lunar_building_bake')
if previous and bpy.app.timers.is_registered(previous):
 raise RuntimeError('Building bake queue is already running')
bpy.app.timers.register(tick,first_interval=2.0)
bpy.app.driver_namespace['lunar_building_bake']=tick
