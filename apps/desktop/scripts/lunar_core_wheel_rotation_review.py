"""Evaluate wheel tread ground contact through a full revolution in Blender."""
import bpy,math,json
from pathlib import Path
from mathutils import Vector
scene=bpy.context.scene
wheels=[o for o in scene.objects if o.type=='EMPTY' and o.name.startswith('wheel.')]
folder=Path(__file__).resolve().parents[1]/'public/lunar-city/worker-multiview-2026-09-07/core-runtime-ux-repairs'
rows=[]
for angle in range(0,361,5):
 for w in wheels:w.rotation_euler.x=math.radians(angle)
 bpy.context.view_layer.update()
 for wheel in wheels:
  points=[o.matrix_world@v.co for o in wheel.children if o.type=='MESH' for v in o.data.vertices]
  rows.append({'wheel':wheel.name,'angle':angle,'minimumZ':min(p.z for p in points)})
for w in wheels:w.rotation_euler.x=0
bpy.context.view_layer.update()
(folder/'wheel-rotation-review.json').write_text(json.dumps({'stage':'study only; interfaces and collision clearance not accepted','samples':len(rows),'minimumZ':min(r['minimumZ'] for r in rows),'wheelCount':len(wheels),'rows':rows},indent=2)+'\n')
print('Wheel rotation minimum Z:',min(r['minimumZ'] for r in rows))
