"""Local card-matched probe repair; preserve all source and v2 worker geometry."""
import bpy,bmesh,json,math,hashlib
from pathlib import Path
from mathutils import Vector
folder=Path(__file__).resolve().parents[1]/'public/lunar-city/worker-multiview-2026-09-07/core-runtime-ux-repairs/static-finish-v3'
folder.mkdir(exist_ok=True);assert not (folder/'complete.json').exists()
source=bpy.data.scenes['Worker static finish | core v2'];scene=bpy.data.scenes.new('Worker static finish | core v3 probe');scene.world=source.world.copy();scene.unit_settings.system='METRIC';scene.unit_settings.scale_length=1
for old in source.objects:
 new=old.copy();new.data=old.data.copy() if old.data else None;new.animation_data_clear();scene.collection.objects.link(new)
 if old==source.camera:scene.camera=new
bpy.context.window.scene=scene
body=next(o for o in scene.objects if o.name.startswith('Finish v2 | Core wheel partition'))
bm=bmesh.new();bm.from_mesh(body.data)
remove=[v for v in bm.verts if (body.matrix_world@v.co).x>.46 and (body.matrix_world@v.co).y<-.28]
assert 100<len(remove)<1200,len(remove)
removed=len(remove);bmesh.ops.delete(bm,geom=remove,context='VERTS');bm.to_mesh(body.data);bm.free()
def material(name,color,metal,rough,emit=0):
 m=bpy.data.materials.new(name);m.use_nodes=True;s=m.node_tree.nodes.get('Principled BSDF');s.inputs['Base Color'].default_value=(*color,1);s.inputs['Metallic'].default_value=metal;s.inputs['Roughness'].default_value=rough
 if emit:s.inputs['Emission Color'].default_value=(*color,1);s.inputs['Emission Strength'].default_value=emit
 return m
steel=material('Probe brushed alloy',(.31,.35,.38),.75,.35);violet=material('Probe violet joint',(.25,.18,.38),.4,.4);graphite=material('Probe grip graphite',(.028,.035,.04),.25,.5);cyan=material('Probe cyan contact',(.02,.7,.8),.1,.24,.8)
parts=[]
def cylinder(name,a,b,r,mat):
 a=Vector(a);b=Vector(b);d=b-a
 bpy.ops.mesh.primitive_cylinder_add(vertices=24,radius=r,depth=d.length,location=(a+b)/2)
 o=bpy.context.object;o.name='Core probe v3 | '+name;o.rotation_euler=d.to_track_quat('Z','Y').to_euler();o.data.materials.append(mat)
 bevel=o.modifiers.new('Machined rim','BEVEL');bevel.width=.0015;bevel.segments=2
 for p in o.data.polygons:p.use_smooth=True
 parts.append(o)
 return o
base=Vector((.545,-.27,.397));axis=(Vector((.635,-.61,.23))-base).normalized();u=axis.cross(Vector((0,0,1))).normalized();v=axis.cross(u).normalized()
def point(t,rad=0,angle=0):return base+axis*t+(u*math.cos(angle)+v*math.sin(angle))*rad
cylinder('wrist socket',point(-.02),point(.04),.044,violet)
cylinder('grip barrel',point(.025),point(.13),.027,graphite)
for t in (.045,.115):cylinder('grip collar',point(t),point(t+.014),.032,steel)
cylinder('central stylus',point(.13),point(.34),.007,steel)
cylinder('cyan sensing tip',point(.34),point(.355),.0075,cyan)
# Three separate finger assemblies leave visible negative space around the stylus.
for i in range(3):
 angle=i*2*math.pi/3
 cylinder(f'finger{i+1} upper',point(.09,.035,angle),point(.17,.044,angle),.009,steel)
 cylinder(f'finger{i+1} knuckle',point(.162,.044,angle),point(.182,.044,angle),.013,violet)
 cylinder(f'finger{i+1} lower',point(.177,.044,angle),point(.235,.014,angle),.007,steel)
for o in scene.objects:o.select_set(o.type=='MESH')
bpy.context.view_layer.objects.active=body
bpy.ops.export_scene.gltf(filepath=str(folder/'static-finish.glb'),export_format='GLB',use_selection=True,use_active_scene=True,export_animations=False)
scene.render.engine='CYCLES';scene.cycles.samples=16;scene.render.resolution_x=800;scene.render.resolution_y=800;scene.render.resolution_percentage=100
cam=scene.camera;cam.data.type='ORTHO';cam.data.ortho_scale=1.55
for name,pos in [('front',(0,-3,.66)),('rear',(0,3,.66)),('left',(-3,0,.66)),('right',(3,0,.66)),('top',(0,0,3)),('overview',(2,-3,1.65))]:
 cam.location=pos;cam.rotation_euler=(Vector((0,0,.60))-cam.location).to_track_quat('-Z','Y').to_euler();scene.render.filepath=str(folder/(name+'.png'));bpy.ops.render.render(write_still=True)
points=[o.matrix_world@v.co for o in scene.objects if o.type=='MESH' for v in o.data.vertices]
receipt={'sourceScene':source.name,'scene':scene.name,'removedDistalToolVertices':removed,'repair':'Three discrete articulated fingers and central cyan-tipped probe, based on all-view card','preservation':'v2 scene and all original source scenes retained','newParts':[o.name for o in parts],'heightMetres':max(p.z for p in points)-min(p.z for p in points),'groundMin':min(p.z for p in points),'sha256':hashlib.sha256((folder/'static-finish.glb').read_bytes()).hexdigest(),'stage':'Static probe candidate; arm/socket and rig acceptance pending'}
assert abs(receipt['heightMetres']-1.2)<.001
(folder/'complete.json').write_text(json.dumps(receipt,indent=2)+'\n')
bpy.ops.wm.save_as_mainfile(filepath=str(folder.parents[2]/'multiview-2026-09-07/leader-rig-workshop.blend'))
print('Core probe v3 candidate saved')
