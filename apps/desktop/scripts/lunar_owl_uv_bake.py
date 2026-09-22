"""Derive a UV-textured review mesh; retain the detailed source scenes."""
from pathlib import Path
import json,math
import bpy
from mathutils import Vector
F=Path(__file__).resolve().parents[1]/'public/lunar-city/multiview-2026-09-07/owl-librarian'
source_scene=bpy.data.scenes['Owl | source color alignment.001']
scene=bpy.data.scenes.new('Owl | UV cleanup review');bpy.context.window.scene=scene
scene.unit_settings.system='METRIC';scene.unit_settings.scale_length=1
source=next(o for o in source_scene.objects if o.type=='MESH' and 'projected' in o.name)
obj=source.copy();obj.data=source.data.copy();obj.parent=None;obj.matrix_world=source.matrix_world.copy();scene.collection.objects.link(obj);obj.name='Owl | UV body review'
bpy.context.view_layer.objects.active=obj;obj.select_set(True)
for modifier in list(obj.modifiers):bpy.ops.object.modifier_apply(modifier=modifier.name)
dec=obj.modifiers.new('Detail reduction for UV review','DECIMATE');dec.ratio=.11
bpy.ops.object.modifier_apply(modifier=dec.name)
bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
bpy.ops.object.mode_set(mode='EDIT');bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.uv.smart_project(angle_limit=math.radians(66),island_margin=.008)
bpy.ops.object.mode_set(mode='OBJECT')
mat=source.data.materials[0].copy();obj.data.materials.clear();obj.data.materials.append(mat)
nodes=mat.node_tree.nodes;links=mat.node_tree.links
color=next(n for n in nodes if n.type=='VERTEX_COLOR');out=next(n for n in nodes if n.type=='OUTPUT_MATERIAL')
emit=nodes.new('ShaderNodeEmission');links.new(color.outputs['Color'],emit.inputs['Color']);links.new(emit.outputs[0],out.inputs['Surface'])
image=bpy.data.images.new('Owl source-color atlas - review',width=2048,height=2048,alpha=False)
image.colorspace_settings.name='sRGB'
texture=nodes.new('ShaderNodeTexImage');texture.image=image;nodes.active=texture
scene.render.engine='CYCLES';scene.cycles.samples=1
scene.render.bake.margin=12
bpy.ops.object.bake(type='EMIT')
image.filepath_raw=str(F/'owl-basecolor-review.png');image.file_format='PNG';image.save();image.pack()
shader=next(n for n in nodes if n.type=='BSDF_PRINCIPLED');links.new(texture.outputs['Color'],shader.inputs['Base Color']);links.new(shader.outputs[0],out.inputs['Surface'])
for old in source_scene.objects:
 if old.type in ('CAMERA','LIGHT') or old.name.startswith('Owl Iris') or old.name.startswith('Owl Pupil'):
  new=old.copy();new.data=old.data.copy();scene.collection.objects.link(new)
  if old==source_scene.camera:scene.camera=new
  if new.type=='MESH':
   new.location.z-=.02;new.location.x-=.012
   if 'Pupil' in new.name:new.scale.y=.022
scene.world=source_scene.world
scene.render.resolution_x=960;scene.render.resolution_y=960;scene.render.resolution_percentage=100
scene.cycles.samples=24;scene.cycles.use_denoising=True
scene.view_settings.view_transform='AgX'
scene.render.filepath=str(F/'uv-review.png')
bpy.ops.object.select_all(action='DESELECT')
for o in scene.objects:
 if o.type=='MESH':o.select_set(True)
bpy.context.view_layer.objects.active=obj
bpy.ops.export_scene.gltf(filepath=str(F/'owl-uv-review.glb'),export_format='GLB',use_selection=True,use_active_scene=True,export_animations=False)
(F/'uv-review.json').write_text(json.dumps({'status':'UV_and_color_bake_review_not_final_rig',
 'bodyFaces':len(obj.data.polygons),'bodyVertices':len(obj.data.vertices),'textureSize':2048,
 'geometryMethod':'0.11 decimation for review; not deformation retopology',
 'textureMethod':'source-projected vertex color baked to UV; alignment remains provisional'},indent=2)+'\n')
bpy.ops.wm.save_as_mainfile(filepath=str(F.parent/'leader-workshop.blend'))
bpy.ops.render.render(write_still=True)
