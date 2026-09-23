"""Four static views for evaluating geometry and paint independently of rigs."""
import bpy,json
from pathlib import Path
from mathutils import Vector
scene=bpy.context.scene;scene.frame_set(0)
folder=Path(__file__).resolve().parents[1]/'public/lunar-city/worker-multiview-2026-09-07/core-runtime-ux-repairs'
cam=scene.camera;old=cam.matrix_world.copy();oldscale=cam.data.ortho_scale
scene.render.engine='CYCLES';scene.cycles.samples=16
scene.render.resolution_x=800;scene.render.resolution_y=800;scene.render.resolution_percentage=100
cam.data.ortho_scale=1.55
for name,pos in [('front',(0,-3,.68)),('rear',(0,3,.68)),('side',(-3,.02,.68)),('top',(0,.02,3))]:
 cam.location=pos;cam.rotation_euler=(Vector((0,.02,.60))-cam.location).to_track_quat('-Z','Y').to_euler()
 scene.render.filepath=str(folder/('static-model-'+name+'.png'));bpy.ops.render.render(write_still=True)
cam.matrix_world=old;cam.data.ortho_scale=oldscale
bpy.ops.wm.save_as_mainfile(filepath=str(folder.parents[1]/'multiview-2026-09-07/leader-rig-workshop.blend'))
print('Static four-view review saved')
