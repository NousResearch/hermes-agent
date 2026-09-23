import bpy,json,hashlib
from pathlib import Path
from mathutils import Vector
folder=Path(__file__).resolve().parents[1]/'public/lunar-city/worker-multiview-2026-09-07/core-runtime-ux-repairs/static-finish-v2'
source=bpy.context.scene;scene=bpy.data.scenes.new('Worker GLB verification | core v2');scene.world=source.world.copy();scene.unit_settings.system='METRIC';scene.unit_settings.scale_length=1;bpy.context.window.scene=scene
bpy.ops.import_scene.gltf(filepath=str(folder/'static-finish.glb'))
meshes=[o for o in scene.objects if o.type=='MESH'];points=[o.matrix_world@v.co for o in meshes for v in o.data.vertices]
height=max(p.z for p in points)-min(p.z for p in points);assert abs(height-1.2)<.001,(height,len(meshes))
assert len(meshes)==9,len(meshes)
for o in source.objects:
 if o.type in {'LIGHT','CAMERA'}:
  copy=o.copy();copy.data=o.data.copy();scene.collection.objects.link(copy)
  if o==source.camera:scene.camera=copy
scene.render.engine='CYCLES';scene.cycles.samples=16;scene.render.resolution_x=800;scene.render.resolution_y=800;scene.render.resolution_percentage=100
cam=scene.camera;cam.location=(2,-3,1.65);cam.rotation_euler=(Vector((0,0,.6))-cam.location).to_track_quat('-Z','Y').to_euler();scene.render.filepath=str(folder/'glb-reimport.png');bpy.ops.render.render(write_still=True)
(folder/'reimport.json').write_text(json.dumps({'meshes':len(meshes),'heightMetres':height,'groundMin':min(p.z for p in points),'sha256':hashlib.sha256((folder/'static-finish.glb').read_bytes()).hexdigest(),'stage':'Static GLB roundtrip, no rig acceptance'},indent=2)+'\n')
bpy.context.window.scene=source;bpy.ops.wm.save_as_mainfile(filepath=str(folder.parents[2]/'multiview-2026-09-07/leader-rig-workshop.blend'))
print('Core static GLB verified',height)
