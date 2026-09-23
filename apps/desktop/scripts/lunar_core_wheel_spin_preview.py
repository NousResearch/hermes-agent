"""Visible rigid rolling study; horizontal travel remains owned by the game."""
import bpy,math,json
from pathlib import Path
from bpy_extras.anim_utils import action_get_channelbag_for_slot
scene=bpy.context.scene
wheels=[o for o in scene.objects if o.type=='EMPTY' and o.name.startswith('wheel.')]
for wheel in wheels:
 wheel.rotation_mode='XYZ';wheel.animation_data_clear()
 for frame,angle in [(0,0),(30,2*math.pi)]:
  wheel.rotation_euler.x=angle;wheel.keyframe_insert('rotation_euler',frame=frame,index=0)
 animation=wheel.animation_data
 animation.action.name='wheel rolling study | '+wheel.name
 bag=action_get_channelbag_for_slot(animation.action,animation.action_slot)
 for curve in bag.fcurves:
  for key in curve.keyframe_points:key.interpolation='LINEAR'
  curve.modifiers.new('CYCLES')
scene.render.fps=30;scene.frame_start=0;scene.frame_end=29;scene.frame_set(0)
root=Path(__file__).resolve().parents[1]/'public/lunar-city'
folder=root/'worker-multiview-2026-09-07/core-runtime-ux-repairs'
(folder/'wheel-spin-study.json').write_text(json.dumps({'stage':'Visible study only; no character export or promotion','wheelRadiusMetres':.12,'cycleSeconds':1,'circumferenceMetres':2*math.pi*.12,'gameSpeedRatioFormula':'speedMetresPerSecond * cycleSeconds / circumferenceMetres','horizontalRootMotion':False,'wheels':[w.name for w in wheels]},indent=2)+'\n')
bpy.ops.wm.save_as_mainfile(filepath=str(root/'multiview-2026-09-07/leader-rig-workshop.blend'))
