"""Review the baked building with a distinct stone apron and portable materials."""
import bpy,json
from pathlib import Path
from mathutils import Vector
ROOT=Path(globals().get('ASSET_ROOT',Path(__file__).resolve().parents[1]/'public/lunar-city/building-multiview-2026-09-07'))
WORKSHOP=Path(__file__).resolve().parents[1]/'public/lunar-city/multiview-2026-09-07/leader-rig-workshop.blend'
asset=ASSET;folder=ROOT/asset;receipt=json.loads((folder/'generation.json').read_text());height=receipt['targetHeightMetres']
scene=bpy.data.scenes.new('Building finish | '+asset);bpy.context.window.scene=scene
scene.unit_settings.system='METRIC';scene.unit_settings.scale_length=1
before=set(bpy.data.objects);bpy.ops.import_scene.gltf(filepath=str(folder/'material-review.glb'))
body=next(o for o in bpy.data.objects if o not in before and o.type=='MESH')
matrix=body.matrix_world.copy();body.parent=None;body.matrix_world=matrix
bpy.context.view_layer.objects.active=body;bpy.ops.object.transform_apply(location=True,rotation=True,scale=True)
body.name=asset+' | building material review'
for material in body.data.materials:
 for node in material.node_tree.nodes:
  if node.type=='NORMAL_MAP':node.inputs['Strength'].default_value=.3
# The reconstructed base plate has no matching elevation artwork. Give only its
# near-horizontal ground faces a stone surface; preserve walls, steps and pools.
stone=bpy.data.materials.new(asset+' | apron stone');stone.use_nodes=True
shader=stone.node_tree.nodes.get('Principled BSDF');shader.inputs['Base Color'].default_value=(.24,.21,.17,1);shader.inputs['Roughness'].default_value=.86
body.data.materials.append(stone);stone_index=len(body.data.materials)-1
apron_faces=0
for poly in body.data.polygons:
 vertices=[body.data.vertices[i].co for i in poly.vertices]
 if max(v.z for v in vertices)<height*.02 and abs(poly.normal.z)>.85:
  poly.material_index=stone_index;apron_faces+=1
bpy.ops.object.select_all(action='DESELECT');body.select_set(True);bpy.context.view_layer.objects.active=body
out=folder/'building-review.glb'
bpy.ops.export_scene.gltf(filepath=str(out),export_format='GLB',use_selection=True,use_active_scene=True,export_animations=False)
body.hide_render=True;body.hide_set(True)
before=set(bpy.data.objects);bpy.ops.import_scene.gltf(filepath=str(out))
meshes=[o for o in bpy.data.objects if o not in before and o.type=='MESH'];bpy.context.view_layer.update()
points=[o.matrix_world@Vector(p) for o in meshes for p in o.bound_box]
bounds=[[min(p[i] for p in points) for i in range(3)],[max(p[i] for p in points) for i in range(3)]]
actual=bounds[1][2]-bounds[0][2]
if abs(actual-height)>.005:raise ValueError(f'Building height {actual} does not match {height}')
span=max(bounds[1][i]-bounds[0][i] for i in range(3))
bpy.ops.object.camera_add();camera=bpy.context.object;camera.data.type='ORTHO';camera.data.ortho_scale=span*1.25;scene.camera=camera
for position,power in [((-2,-3,3),450),((2,-2,2),180),((1,2,3),350)]:
 bpy.ops.object.light_add(type='AREA',location=Vector(position)*height/1.5);light=bpy.context.object;light.data.energy=power*(height/1.5)**2;light.data.size=height*2
 light.rotation_euler=(Vector((0,0,height*.5))-light.location).to_track_quat('-Z','Y').to_euler()
scene.world=bpy.data.worlds.new(asset+' building review world');scene.world.use_nodes=True
scene.world.node_tree.nodes['Background'].inputs[0].default_value=(.09,.11,.13,1);scene.world.node_tree.nodes['Background'].inputs[1].default_value=.35
scene.render.engine='CYCLES';scene.cycles.samples=24;scene.cycles.use_denoising=True;scene.view_settings.view_transform='AgX'
scene.render.resolution_x=960;scene.render.resolution_y=960;scene.render.resolution_percentage=100
for label,position in [('front',(height*.2,-span*3,height*1.15)),('rear',(0,span*3,height*.9)),('side',(-span*3,0,height*.9)),('top',(0,0,span*4))]:
 camera.location=position;camera.rotation_euler=(Vector((0,0,height*.5))-camera.location).to_track_quat('-Z','Y').to_euler();scene.render.filepath=str(folder/('building-'+label+'.png'));bpy.ops.render.render(write_still=True)
camera.location=(height*.2,-span*3,height*1.15);camera.rotation_euler=(Vector((0,0,height*.5))-camera.location).to_track_quat('-Z','Y').to_euler()
(folder/'building-finish-review.json').write_text(json.dumps({'asset':asset,'stage':'material_review_requires_visual_and_runtime_acceptance','heightMetres':actual,'apronFaces':apron_faces,'blenderBounds':bounds,'exportAxes':'Y up, +Z front','entrancePlacement':'requires inspection against source steps; do not infer from slab edge'},indent=2)+'\n')
for area in bpy.context.screen.areas:
 if area.type=='CONSOLE':area.type='VIEW_3D'
 if area.type=='VIEW_3D':
  area.spaces.active.region_3d.view_perspective='CAMERA';area.spaces.active.shading.type='MATERIAL'
bpy.ops.wm.save_as_mainfile(filepath=str(WORKSHOP))
