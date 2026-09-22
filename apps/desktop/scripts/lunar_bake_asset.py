"""Finite visible batch of UV/color/normal review assets, not finished rigs."""
from pathlib import Path
import json,math,traceback
import bpy
from mathutils import Vector
ROOT=Path(ASSET_ROOT)
WORKSHOP=Path(__file__).resolve().parents[1]/'public/lunar-city/multiview-2026-09-07/leader-rig-workshop.blend'
ASSETS=['elephant-memory','cat-arts','fox-scientist','capybara-revenue','lion-steward','beaver-architect','monkey-poet']

def process(asset):
 folder=ROOT/asset;receipt=json.loads((folder/'generation.json').read_text());height=receipt['targetHeightMetres']
 scene=bpy.data.scenes.new('Materials | '+asset);bpy.context.window.scene=scene
 scene.unit_settings.system='METRIC';scene.unit_settings.scale_length=1
 before=set(bpy.data.objects);bpy.ops.import_scene.gltf(filepath=str(folder/'source-color-study-v4.glb'))
 high=next(o for o in bpy.data.objects if o not in before and o.type=='MESH')
 world=high.matrix_world.copy();high.parent=None;high.matrix_world=world
 high.name=asset+' | detailed color source'
 for poly in high.data.polygons:poly.use_smooth=True
 smooth=high.modifiers.new('Surface smoothing','SMOOTH');smooth.factor=.18;smooth.iterations=2
 mat=bpy.data.materials.new(asset+' | source bake');mat.use_nodes=True
 nodes=mat.node_tree.nodes;links=mat.node_tree.links
 color=nodes.new('ShaderNodeVertexColor');color.layer_name=high.data.color_attributes[0].name
 emit=nodes.new('ShaderNodeEmission');links.new(color.outputs['Color'],emit.inputs['Color']);links.new(emit.outputs[0],nodes.get('Material Output').inputs['Surface'])
 high.data.materials.clear();high.data.materials.append(mat)
 low=high.copy();low.data=high.data.copy();scene.collection.objects.link(low);low.name=asset+' | textured review'
 bpy.ops.object.select_all(action='DESELECT');low.select_set(True);bpy.context.view_layer.objects.active=low
 for modifier in list(low.modifiers):bpy.ops.object.modifier_apply(modifier=modifier.name)
 dec=low.modifiers.new('Runtime density study','DECIMATE');dec.ratio=min(1,globals().get('FACE_BUDGET',30000)/len(low.data.polygons));bpy.ops.object.modifier_apply(modifier=dec.name)
 bpy.ops.object.transform_apply(location=False,rotation=True,scale=True)
 bpy.ops.object.mode_set(mode='EDIT');bpy.ops.mesh.select_all(action='SELECT');bpy.ops.uv.smart_project(angle_limit=math.radians(66),island_margin=.008);bpy.ops.object.mode_set(mode='OBJECT')
 target=bpy.data.materials.new(asset+' | baked review surface');target.use_nodes=True;low.data.materials.clear();low.data.materials.append(target)
 tn=target.node_tree.nodes;tl=target.node_tree.links;shader=tn.get('Principled BSDF');shader.inputs['Roughness'].default_value=.65
 albedo=bpy.data.images.new(asset+' | basecolor',width=2048,height=2048,alpha=False);albedo.colorspace_settings.name='sRGB'
 at=tn.new('ShaderNodeTexImage');at.image=albedo;tn.active=at
 scene.render.engine='CYCLES';scene.cycles.samples=1;scene.render.bake.margin=12
 scene.render.bake.use_selected_to_active=True;scene.render.bake.cage_extrusion=height*.01;scene.render.bake.max_ray_distance=height*.025
 high.select_set(True);bpy.context.view_layer.objects.active=low
 bpy.ops.object.bake(type='EMIT')
 albedo.filepath_raw=str(folder/'basecolor-review.png');albedo.file_format='PNG';albedo.save();albedo.pack();tl.new(at.outputs['Color'],shader.inputs['Base Color'])
 normal=bpy.data.images.new(asset+' | normal',width=2048,height=2048,alpha=False);normal.colorspace_settings.name='Non-Color'
 nt=tn.new('ShaderNodeTexImage');nt.image=normal;tn.active=nt
 bpy.ops.object.bake(type='NORMAL')
 normal.filepath_raw=str(folder/'normal-review.png');normal.file_format='PNG';normal.save();normal.pack()
 nm=tn.new('ShaderNodeNormalMap');nm.uv_map=low.data.uv_layers.active.name;tl.new(nt.outputs['Color'],nm.inputs['Color']);tl.new(nm.outputs['Normal'],shader.inputs['Normal'])
 high.hide_render=True;high.hide_set(True);scene.render.bake.use_selected_to_active=False
 bpy.ops.object.select_all(action='DESELECT');low.select_set(True);bpy.context.view_layer.objects.active=low
 # Smoothing/decimation can shave extrema. Restore the explicit metre contract
 # uniformly after baking, without changing the original source mesh.
 bpy.ops.object.transform_apply(location=True,rotation=True,scale=True)
 min_z=min(v.co.z for v in low.data.vertices);max_z=max(v.co.z for v in low.data.vertices)
 restoration_scale=height/(max_z-min_z)
 for vertex in low.data.vertices:
  vertex.co.z-=min_z;vertex.co*=restoration_scale
 low.data.update();bpy.context.view_layer.update()
 bpy.ops.export_scene.gltf(filepath=str(folder/'material-review.glb'),export_format='GLB',use_selection=True,use_active_scene=True,export_animations=False)
 low.hide_render=True;low.hide_set(True)
 before=set(bpy.data.objects);bpy.ops.import_scene.gltf(filepath=str(folder/'material-review.glb'))
 reimported=[o for o in bpy.data.objects if o not in before and o.type=='MESH'];bpy.context.view_layer.update()
 points=[o.matrix_world@Vector(v) for o in reimported for v in o.bound_box]
 actual=max(v.z for v in points)-min(v.z for v in points)
 if abs(actual-height)>.005:raise ValueError(f'{asset}: metric height changed {height} -> {actual}')
 if len(reimported)!=1:raise ValueError(f'{asset}: unexpected export objects')
 for obj in reimported:obj.name=asset+' | verified GLB review'
 bpy.ops.object.camera_add(location=(height*.22,-height*2.8,height*1.25));camera=bpy.context.object
 camera.rotation_euler=(Vector((0,0,height*.51))-camera.location).to_track_quat('-Z','Y').to_euler();camera.data.type='ORTHO'
 spans=[max(v[i] for v in points)-min(v[i] for v in points) for i in range(3)]
 camera.data.ortho_scale=max(spans)*1.4;scene.camera=camera
 for name,xyz,power in [('Key',(-2,-3,3),450),('Fill',(2,-2,2),180),('Rim',(1,2,3),350)]:
  pos=Vector(xyz)*height/1.5;bpy.ops.object.light_add(type='AREA',location=pos);o=bpy.context.object;o.name=asset+' '+name;o.data.energy=power*(height/1.5)**2;o.data.shape='DISK';o.data.size=height*2
  o.rotation_euler=(Vector((0,0,height*.5))-o.location).to_track_quat('-Z','Y').to_euler()
 scene.world=bpy.data.worlds.new(asset+' studio');scene.world.use_nodes=True;scene.world.node_tree.nodes['Background'].inputs[0].default_value=(.09,.11,.13,1);scene.world.node_tree.nodes['Background'].inputs[1].default_value=.4
 scene.cycles.samples=24;scene.cycles.use_denoising=True;scene.view_settings.view_transform='AgX'
 scene.render.resolution_x=960;scene.render.resolution_y=960;scene.render.resolution_percentage=100;scene.render.filepath=str(folder/'material-review.png')
 for screen in bpy.data.screens:
  for area in screen.areas:
   if area.type=='VIEW_3D':area.spaces.active.region_3d.view_perspective='CAMERA';area.spaces.active.shading.type='MATERIAL'
 bpy.ops.render.render(write_still=True)
 saved_matrix=camera.matrix_world.copy()
 for direction,position in [('rear',(0,height*3,height*.8)),('side',(-height*3,0,height*.8)),('top',(0,0,height*4))]:
  camera.location=position;camera.rotation_euler=(Vector((0,0,height*.5))-camera.location).to_track_quat('-Z','Y').to_euler()
  scene.render.filepath=str(folder/('material-'+direction+'.png'));bpy.ops.render.render(write_still=True)
 camera.matrix_world=saved_matrix
 (folder/'material-review.json').write_text(json.dumps({'asset':asset,'stage':'baked_material_review_requires_visual_cleanup_and_rig',
  'faces':len(low.data.polygons),'sourceHeightMetres':height,'reimportHeightMetres':actual,'uniformScaleAfterSmoothing':restoration_scale,
  'textures':['basecolor-review.png','normal-review.png'],'textureSize':2048,
  'limitations':['projection alignment remains provisional','no deformation topology or rig','surface material roughness provisional']},indent=2)+'\n')
 bpy.ops.wm.save_as_mainfile(filepath=str(WORKSHOP))


process(ASSET)
