"""Seat wheel faces against sidewalls and bake each assembly into one rigid mesh."""
import bpy,json,math
from pathlib import Path
from mathutils import Matrix,Vector
scene=bpy.context.scene
scene.frame_set(0);bpy.context.view_layer.update()
wheels=[o for o in scene.objects if o.type=='EMPTY' and o.name.startswith('wheel.')]
assert len(wheels)==4
assert all(not w.get('unified_wheel') for w in wheels), 'Already unified; inspect instead of repeating'
backup=bpy.data.collections.new('Preserved separated wheel parts before rim seating')
scene.collection.children.link(backup);backup.hide_render=True;backup.hide_viewport=True
reports=[];assemblies=[]
for w in wheels:
 parts=[o for o in w.children if o.type=='MESH'];assert len(parts)>20
 verts=[];faces=[];matids=[];smooth=[];materials=[]
 sign=1 if w.location.x>0 else -1
 for o in parts:
  m=w.matrix_world.inverted()@o.matrix_world
  label=o.name.split(' | ',1)[1]
  # Pull the metal face into the rubber bead; preserve tire radius and axle.
  shift=sign*(-.022) if not w.get('rim_seated') and any(label.startswith(t) for t in ('outer hub','rim','cap','axle cap','hub bolt','indicator')) else 0
  start=len(verts)
  for v in o.data.vertices:
   p=m@v.co;p.x+=shift
   if label=='hub barrel' and not w.get('rim_seated'):p.x*=.6875
   verts.append(p)
  for poly in o.data.polygons:
   material=o.data.materials[poly.material_index]
   if material not in materials:materials.append(material)
   faces.append([start+i for i in poly.vertices]);matids.append(materials.index(material));smooth.append(poly.use_smooth)
 mesh=bpy.data.meshes.new(w.name+' seated wheel mesh');mesh.from_pydata(verts,[],faces);mesh.update()
 for mat in materials:mesh.materials.append(mat)
 for p,mi,sm in zip(mesh.polygons,matids,smooth):p.material_index=mi;p.use_smooth=sm
 obj=bpy.data.objects.new(w.name+' | unified tire and rim',mesh);scene.collection.objects.link(obj)
 obj.parent=w;obj.matrix_parent_inverse=Matrix.Identity(4);obj.matrix_basis=Matrix.Identity(4)
 for o in parts:
  world=o.matrix_world.copy();o.parent=None;o.matrix_world=world
  for c in list(o.users_collection):c.objects.unlink(o)
  backup.objects.link(o)
 w['unified_wheel']=True;w['rim_seated']=True;assemblies.append((w,obj))
 reports.append({'wheel':w.name,'sourceParts':len(parts),'activeMeshes':1,'vertices':len(verts),'rimAxialCorrectionMetres':-.022,'radiusMetres':.12})
bpy.context.view_layer.update()
max_error=0;min_z=1e9
for f in range(31):
 scene.frame_set(f);dg=bpy.context.evaluated_depsgraph_get()
 for w,o in assemblies:
  ev=o.evaluated_get(dg);me=ev.to_mesh()
  try:
   for base,v in zip(o.data.vertices,me.vertices):
    actual=ev.matrix_world@v.co;expected=w.matrix_world@base.co
    max_error=max(max_error,(actual-expected).length);min_z=min(min_z,actual.z)
  finally:ev.to_mesh_clear()
assert max_error<1e-6,(max_error,'relative wheel deformation')
assert min_z>=-1e-5,(min_z,'ground penetration')
scene.frame_set(0)
root=Path(__file__).resolve().parents[1]/'public/lunar-city'
folder=root/'worker-multiview-2026-09-07/core-runtime-ux-repairs'
(folder/'wheel-unified-review.json').write_text(json.dumps({'stage':'Rigid wheel correction; full character remains a study','wheels':reports,'evaluatedFrames':31,'maximumRigidErrorMetres':max_error,'minimumGroundMetres':min_z,'preservedCollection':backup.name},indent=2)+'\n')
bpy.ops.wm.save_as_mainfile(filepath=str(root/'multiview-2026-09-07/leader-rig-workshop.blend'))
print('Unified wheels saved; max rigid error:',max_error,'ground:',min_z)
