"""Non-production wheel face study against the recovered core-runtime mesh."""
import bpy,json,math
from pathlib import Path
from mathutils import Vector
root=Path(__file__).resolve().parents[1]/'public/lunar-city'
folder=root/'worker-multiview-2026-09-07/core-runtime-ux-repairs'
scene=bpy.data.scenes.new('Mechanical study | core wheels');bpy.context.window.scene=scene
scene.unit_settings.system='METRIC';scene.unit_settings.scale_length=1
before=set(bpy.data.objects);bpy.ops.import_scene.gltf(filepath=str(folder/'material-review.glb'))
body=max((o for o in bpy.data.objects if o not in before and o.type=='MESH'),key=lambda o:len(o.data.vertices))
bpy.context.view_layer.objects.active=body;bpy.ops.object.transform_apply(location=False,rotation=True,scale=True)
body.name='Core wheel partition study | source geometry unchanged'
source_materials=list(body.data.materials)
body.data.materials.clear()
for name,color in [('Stationary',(0.32,.34,.38,1)),('Front wheels',(.9,.23,.045,1)),('Rear wheels',(.045,.60,.24,1))]:
 m=bpy.data.materials.new(name);m.diffuse_color=color;m.use_nodes=True;m.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value=color;m.node_tree.nodes['Principled BSDF'].inputs['Roughness'].default_value=.65;body.data.materials.append(m)
regions=[{'name':end+'.'+side,'center':[sign*.25,depth,.12],'radius':.13,'axle':[1,0,0],'material':mat,'faces':[]} for side,sign in [('L',1),('R',-1)] for end,depth,mat in [('front',-.17,1),('rear',.30,2)]]
for face in body.data.polygons:
 p=face.center
 for region in regions:
  cx,cy,cz=region['center']
  if p.x*math.copysign(1,cx)>.18 and abs(p.x)<.34 and math.hypot(p.y-cy,p.z-cz)<region['radius']:
   face.material_index=region['material'];region['faces'].append(face.index);break
bpy.ops.object.camera_add(location=(-3,.06,.35));cam=bpy.context.object;cam.rotation_euler=(Vector((0,.06,.30))-cam.location).to_track_quat('-Z','Y').to_euler();cam.data.type='ORTHO';cam.data.ortho_scale=.9;scene.camera=cam
for pos,power in [((-2,-2,3),400),((1,1,2),300)]:
 bpy.ops.object.light_add(type='AREA',location=pos);o=bpy.context.object;o.data.energy=power;o.data.size=3;o.rotation_euler=(Vector((0,0,.5))-o.location).to_track_quat('-Z','Y').to_euler()
scene.world=bpy.data.worlds.new('Mechanical study world');scene.world.use_nodes=True;scene.world.node_tree.nodes['Background'].inputs[0].default_value=(.1,.12,.15,1);scene.world.node_tree.nodes['Background'].inputs[1].default_value=.4
scene.render.engine='CYCLES';scene.cycles.samples=24;scene.cycles.use_denoising=True;scene.render.resolution_x=960;scene.render.resolution_y=720;scene.render.resolution_percentage=100
scene.render.filepath=str(folder/'wheel-partition-study-side.png');bpy.ops.render.render(write_still=True)
(folder/'wheel-partition-study.json').write_text(json.dumps({'stage':'candidate face regions; geometry unchanged; no production promotion','units':'metres','regions':regions},indent=2)+'\n')
bpy.ops.wm.save_as_mainfile(filepath=str(root/'multiview-2026-09-07/leader-rig-workshop.blend'))
print([(r['name'],len(r['faces'])) for r in regions])
