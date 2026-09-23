"""Preserved-source study of rigid wheel replacements, not a promoted asset."""
import bpy,bmesh,runpy,math,json
from pathlib import Path
from mathutils import Vector
base=Path(__file__).resolve().parent
context=runpy.run_path(str(base/'lunar_core_wheel_partition_study.py'))
scene=context['scene'];body=context['body'];folder=context['folder'];root=context['root'];regions=context['regions']
scene.name='Mechanical study | core rigid wheels'
old=body.data;old.use_fake_user=True;body.data=old.copy()
bm=bmesh.new();bm.from_mesh(body.data);bmesh.ops.delete(bm,geom=[f for f in bm.faces if f.material_index>0],context='FACES');bm.to_mesh(body.data);bm.free();body.data.update()
body.data.materials.clear()
for m in context['source_materials']:body.data.materials.append(m)
for face in body.data.polygons:face.material_index=0
def material(name,color,metal=0,rough=.5,emission=0):
 m=bpy.data.materials.new(name);m.diffuse_color=(*color,1);m.use_nodes=True;p=m.node_tree.nodes['Principled BSDF'];p.inputs['Base Color'].default_value=(*color,1);p.inputs['Metallic'].default_value=metal;p.inputs['Roughness'].default_value=rough
 if emission:p.inputs['Emission Color'].default_value=(*color,1);p.inputs['Emission Strength'].default_value=emission
 return m
rubber=material('Core tyre rubber',(.027,.031,.038),0,.78)
metal=material('Core machined hub',(.25,.28,.30),.8,.28)
dark=material('Core wheel recess',(.065,.073,.085),.6,.45)
cyan=material('Core hub cyan indicators',(.03,.55,.65),.3,.3,1.2)
wheel_objects=[]
for region in regions:
 center=Vector(region['center']);sign=1 if center.x>0 else -1
 bpy.ops.object.empty_add(type='PLAIN_AXES',location=center);wheel=bpy.context.object;wheel.name='wheel.'+region['name'];parts=[]
 def retain(o,mat,name):
  o.name=wheel.name+' | '+name;o.data.materials.append(mat);parts.append(o)
 def cylinder(radius,depth,offset,mat,name,vertices=48):
  bpy.ops.mesh.primitive_cylinder_add(vertices=vertices,radius=radius,depth=depth,location=center+Vector((sign*offset,0,0)),rotation=(0,math.pi/2,0));retain(bpy.context.object,mat,name)
 def torus(major,minor,offset,mat,name):
  bpy.ops.mesh.primitive_torus_add(major_segments=64,minor_segments=12,major_radius=major,minor_radius=minor,location=center+Vector((sign*offset,0,0)),rotation=(0,math.pi/2,0));retain(bpy.context.object,mat,name)
 torus(.09,.025,0,rubber,'tyre')
 cylinder(.079,.044,0,dark,'hub barrel')
 cylinder(.071,.009,.014,metal,'outer hub')
 torus(.068,.005,.021,metal,'rim')
 cylinder(.036,.012,.021,dark,'cap')
 cylinder(.025,.006,.030,metal,'axle cap')
 for i in range(24):
  angle=2*math.pi*i/24
  bpy.ops.mesh.primitive_cube_add(size=1,location=center+Vector((0,(math.sqrt(.12**2-.009**2)-.005)*math.cos(angle),(math.sqrt(.12**2-.009**2)-.005)*math.sin(angle))))
  o=bpy.context.object;o.scale=(.055,.010,.018);o.rotation_euler[0]=angle;retain(o,rubber,'tread')
 for i in range(6):
  angle=2*math.pi*i/6
  pos=center+Vector((sign*.025,.050*math.cos(angle),.050*math.sin(angle)))
  bpy.ops.mesh.primitive_cylinder_add(vertices=12,radius=.004,depth=.006,location=pos,rotation=(0,math.pi/2,0));retain(bpy.context.object,metal,'hub bolt')
  pos=center+Vector((sign*.025,.061*math.cos(angle+.25),.061*math.sin(angle+.25)))
  bpy.ops.mesh.primitive_uv_sphere_add(segments=12,ring_count=6,radius=.003,location=pos);retain(bpy.context.object,cyan,'indicator')
 for o in parts:
  matrix=o.matrix_world.copy();o.parent=wheel;o.matrix_world=matrix
  for polygon in o.data.polygons:polygon.use_smooth='tread' not in o.name
 wheel['rim_seated']=True
 wheel_objects.append(wheel)
scene.render.filepath=str(folder/'wheel-retopology-study-side.png');bpy.ops.render.render(write_still=True)
(folder/'wheel-retopology-study.json').write_text(json.dumps({'stage':'Rigid wheel study; source interfaces and stabilizer clearance require review; not production','sourceMeshPreserved':old.name,'wheels':[{'name':w.name,'center':list(w.location),'axle':[1,0,0],'nominalRadiusMetres':.12,'rigidChildren':len(w.children)} for w in wheel_objects]},indent=2)+'\n')
bpy.ops.wm.save_as_mainfile(filepath=str(root/'multiview-2026-09-07/leader-rig-workshop.blend'))
