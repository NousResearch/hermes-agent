"""Source-card visor surface with separate emissive eyes; no whole-head replacement."""
import bpy,json,math,hashlib
from pathlib import Path
from mathutils import Vector
from mathutils.bvhtree import BVHTree
folder=Path(__file__).resolve().parents[1]/'public/lunar-city/worker-multiview-2026-09-07/core-runtime-ux-repairs/static-finish-v3'
scene=bpy.data.scenes['Worker static finish | core v3 probe'];bpy.context.window.scene=scene
for o in list(scene.objects):
 if o.name.startswith(('Core clean visor |','Core cyan eye |')):bpy.data.objects.remove(o,do_unlink=True)
body=next(o for o in scene.objects if o.name.startswith('Finish v2 | Core wheel partition'))
tree=BVHTree.FromObject(body,bpy.context.evaluated_depsgraph_get())
def material(name,color,rough,emit=0):
 m=bpy.data.materials.new(name);m.use_nodes=True;s=m.node_tree.nodes.get('Principled BSDF');s.inputs['Base Color'].default_value=(*color,1);s.inputs['Metallic'].default_value=.15;s.inputs['Roughness'].default_value=rough
 if emit:s.inputs['Emission Color'].default_value=(*color,1);s.inputs['Emission Strength'].default_value=emit
 return m
def panel(name,cx,cz,width,height,radius,offset,mat):
 edge=[]
 for x,z,start in [(width/2-radius,height/2-radius,0),(-width/2+radius,height/2-radius,90),(-width/2+radius,-height/2+radius,180),(width/2-radius,-height/2+radius,270)]:
  for k in range(9):
   a=math.radians(start+k*90/8);edge.append((cx+x+radius*math.cos(a),cz+z+radius*math.sin(a)))
 verts=[]
 for f in [i/24 for i in range(25)]:
  for x,z in edge:
   x=cx+(x-cx)*f;z=cz+(z-cz)*f
   hit,normal,_,_=tree.ray_cast(Vector((x,-1,z)),Vector((0,1,0)),2)
   if hit is None:raise RuntimeError('Visor ray missed front shell at '+str((x,z,hit)))
   verts.append((x,-.255+.30*x*x-offset,z))
 n=len(edge);faces=[]
 for ring in range(24):
  for i in range(n):j=(i+1)%n;faces.append((ring*n+i,ring*n+j,(ring+1)*n+j,(ring+1)*n+i))
 mesh=bpy.data.meshes.new(name);mesh.from_pydata(verts,[],faces);mesh.update();o=bpy.data.objects.new(name,mesh);scene.collection.objects.link(o);mesh.materials.append(mat)
 for p in mesh.polygons:p.use_smooth=True
 return o
panel('Core clean visor | source conforming',0,.825,.38,.17,.045,0,material('Core clean graphite visor',(.008,.018,.025),.22))
for x in [-.075,.075]:panel('Core cyan eye | '+str(x),x,.832,.043,.056,.018,.003,material('Core cyan display '+str(x),(.025,.72,.82),.28,1.1))
scene['visor_finished']=True
for o in scene.objects:o.select_set(o.type=='MESH')
bpy.context.view_layer.objects.active=body
bpy.ops.export_scene.gltf(filepath=str(folder/'static-finish.glb'),export_format='GLB',use_selection=True,use_active_scene=True,export_animations=False)
cam=scene.camera
for name,pos in [('front',(0,-3,.66)),('rear',(0,3,.66)),('left',(-3,0,.66)),('right',(3,0,.66)),('top',(0,0,3)),('overview',(2,-3,1.65))]:
 cam.location=pos;cam.rotation_euler=(Vector((0,0,.60))-cam.location).to_track_quat('-Z','Y').to_euler();scene.render.filepath=str(folder/(name+'.png'));bpy.ops.render.render(write_still=True)
receipt=json.loads((folder/'complete.json').read_text());receipt['visor']='Source-conforming graphite panel, two distinct cyan eyes';receipt['sha256']=hashlib.sha256((folder/'static-finish.glb').read_bytes()).hexdigest();(folder/'complete.json').write_text(json.dumps(receipt,indent=2)+'\n')
bpy.ops.wm.save_as_mainfile(filepath=str(folder.parents[2]/'multiview-2026-09-07/leader-rig-workshop.blend'))
print('Core clean visor saved')
