"""Check rotating tire triangles against the stationary reconstructed chassis."""
import bpy,math,json
from pathlib import Path
from mathutils.bvhtree import BVHTree
scene=bpy.context.scene
body=next(o for o in scene.objects if o.type=='MESH' and o.name.startswith('Core wheel partition study'))
stationary=BVHTree.FromPolygons([body.matrix_world@v.co for v in body.data.vertices],[p.vertices[:] for p in body.data.polygons])
wheels=[o for o in scene.objects if o.type=='EMPTY' and o.name.startswith('wheel.')]
rows=[]
for angle in range(0,360,10):
 for w in wheels:w.rotation_euler.x=math.radians(angle)
 bpy.context.view_layer.update()
 for w in wheels:
  vertices=[];polygons=[]
  for o in w.children:
   if o.type!='MESH' or not any(n in o.name for n in [' | tyre',' | tread']):continue
   offset=len(vertices);vertices.extend(o.matrix_world@v.co for v in o.data.vertices);polygons.extend([[i+offset for i in p.vertices] for p in o.data.polygons])
  tire=BVHTree.FromPolygons(vertices,polygons)
  overlaps=stationary.overlap(tire)
  rows.append({'wheel':w.name,'angle':angle,'triangleIntersections':len(overlaps),'stationaryFaces':sorted({a for a,b in overlaps})})
for w in wheels:w.rotation_euler.x=0
bpy.context.view_layer.update()
folder=Path(__file__).resolve().parents[1]/'public/lunar-city/worker-multiview-2026-09-07/core-runtime-ux-repairs'
summary={'stage':'Study tire/chassis triangle intersection audit; not complete enclosed-volume or final clearance acceptance','samples':len(rows),'maxTriangleIntersections':max(r['triangleIntersections'] for r in rows),'rows':rows}
(folder/'wheel-clearance-review.json').write_text(json.dumps(summary,indent=2)+'\n')
print('Maximum tire/chassis triangle intersections:',summary['maxTriangleIntersections'])
