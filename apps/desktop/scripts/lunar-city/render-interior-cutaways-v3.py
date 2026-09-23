import bpy,json
from pathlib import Path
from mathutils import Vector
out=Path(__file__).resolve().parents[2]/'public/lunar-city/interior-plans-v1'
scene=bpy.data.scenes['Lunar City Interior Studies v3'];bpy.context.window.scene=scene
scene.render.engine='CYCLES';scene.cycles.samples=24
world=bpy.data.worlds.new('Interior neutral studio');world.use_nodes=True;world.node_tree.nodes['Background'].inputs['Color'].default_value=(.38,.42,.46,1);world.node_tree.nodes['Background'].inputs['Strength'].default_value=.8;scene.world=world
for o in scene.objects:
 if o.type=='LIGHT':o.data.energy=50000
scene.render.filepath=str(out/'blender-gallery-v3.png');bpy.ops.render.render(write_still=True)
cam=scene.camera;matrix=cam.matrix_world.copy();scale=cam.data.ortho_scale
plan=json.loads((out/'plans.json').read_text())['plans'];idx=next(i for i,p in enumerate(plan) if p['id']=='engineering-workshop');center=Vector(((idx%4)*28,-(idx//4)*28,0))
cam.location=center+Vector((14,-18,20));cam.rotation_euler=(center+Vector((0,0,1))-cam.location).to_track_quat('-Z','Y').to_euler();cam.data.ortho_scale=24;scene.render.resolution_x=1200;scene.render.resolution_y=1000;scene.render.filepath=str(out/'beaver-cutaway-v3.png');bpy.ops.render.render(write_still=True)
cam.matrix_world=matrix;cam.data.ortho_scale=scale;scene.render.resolution_x=1800;scene.render.resolution_y=1800
bpy.ops.wm.save_as_mainfile(filepath=str(out.parent/'multiview-2026-09-07/leader-rig-workshop.blend'))
bpy.context.area.type='VIEW_3D';bpy.context.space_data.region_3d.view_perspective='CAMERA';bpy.context.space_data.shading.type='MATERIAL'
print('Interior gallery saved and visible')
