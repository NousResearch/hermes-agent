import bpy,runpy,json,traceback
from pathlib import Path
SCRIPTS=Path(__file__).resolve().parent
ROOT=SCRIPTS.parent/'public/lunar-city/multiview-2026-09-07'
assets=iter(['fox-scientist','monkey-poet','capybara-revenue','lion-steward','beaver-architect','owl-librarian','elephant-memory'])
def tick():
 try:asset=next(assets)
 except StopIteration:return None
 try:
  for script,stage in [('lunar_leader_rig.py','fitting_weights'),('lunar_surface_finish.py','surface_finish'),('lunar_animation_export.py','animation_export')]:
   (ROOT/'rig-batch-status.json').write_text(json.dumps({'asset':asset,'stage':stage})+'\n')
   runpy.run_path(str(SCRIPTS/script),init_globals={'ASSET':asset})
  (ROOT/'rig-batch-status.json').write_text(json.dumps({'asset':asset,'stage':'exported_requires_visual_review'})+'\n')
 except Exception:
  (ROOT/'rig-batch-error.txt').write_text(traceback.format_exc());raise
 return 3.0
bpy.app.timers.register(tick,first_interval=2.0)
bpy.app.driver_namespace['lunar_rig_batch']=tick
print('Finite leader rig and surface batch registered.')
