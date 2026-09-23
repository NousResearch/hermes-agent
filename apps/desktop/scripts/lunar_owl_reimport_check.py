from pathlib import Path
import json
import bpy
from mathutils import Vector
F=Path(__file__).resolve().parents[1]/'public/lunar-city/multiview-2026-09-07/owl-librarian'
original=bpy.data.scenes['Owl | UV cleanup review'];bpy.context.window.scene=original
body=next(o for o in original.objects if o.name.startswith('Owl | UV body'))
for n in body.data.materials[0].node_tree.nodes:
 if n.type=='NORMAL_MAP':n.uv_map=body.data.uv_layers.active.name
bpy.ops.object.select_all(action='DESELECT')
for o in original.objects:
 if o.type=='MESH' and not o.hide_get():o.select_set(True)
bpy.ops.export_scene.gltf(filepath=str(F/'owl-detail-review.glb'),export_format='GLB',use_selection=True,use_active_scene=True,export_animations=False)
scene=bpy.data.scenes.new('Owl | exported GLB verification');bpy.context.window.scene=scene
scene.unit_settings.system='METRIC';scene.unit_settings.scale_length=1
bpy.ops.import_scene.gltf(filepath=str(F/'owl-detail-review.glb'))
bpy.context.view_layer.update()
meshes=[o for o in scene.objects if o.type=='MESH']
points=[o.matrix_world@Vector(v) for o in meshes for v in o.bound_box]
height=max(p.z for p in points)-min(p.z for p in points)
assert len(meshes)==5,len(meshes)
assert abs(height-1.5)<.005,height
assert all(o.data.uv_layers for o in meshes)
for old in original.objects:
 if old.type in ('CAMERA','LIGHT'):
  obj=old.copy();obj.data=old.data.copy();scene.collection.objects.link(obj)
  if old==original.camera:scene.camera=obj
scene.world=original.world;scene.render.engine='CYCLES';scene.cycles.samples=32;scene.cycles.use_denoising=True
scene.render.resolution_x=960;scene.render.resolution_y=960;scene.render.resolution_percentage=100
scene.view_settings.view_transform='AgX';scene.render.filepath=str(F/'export-reimport.png')
(F/'export-validation.json').write_text(json.dumps({'meshObjects':len(meshes),'heightMetres':height,'uvPresent':True,'status':'Blender_reimport_passed_game_runtime_and_rig_pending'},indent=2)+'\n')
bpy.ops.wm.save_as_mainfile(filepath=str(F.parent/'leader-workshop.blend'))
bpy.ops.render.render(write_still=True)
