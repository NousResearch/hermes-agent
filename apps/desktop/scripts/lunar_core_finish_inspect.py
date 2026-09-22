import bpy,json
from pathlib import Path
from mathutils import Vector
scene=bpy.context.scene
body=next(o for o in scene.objects if o.name.startswith('Core wheel partition'))
pts=[body.matrix_world@v.co for v in body.data.vertices]
report={'scene':scene.name,'body':body.name,'vertices':len(pts),'polygons':len(body.data.polygons),'bounds':[[min(p[i] for p in pts) for i in range(3)],[max(p[i] for p in pts) for i in range(3)]],'materials':[m.name for m in body.data.materials],'uv':[u.name for u in body.data.uv_layers],'modifiers':[(m.name,m.type) for m in body.modifiers],'visibleObjects':[(o.name,o.type) for o in scene.objects if o.visible_get() and not o.hide_render]}
Path('/private/tmp/lunar-core-finish-inspect.json').write_text(json.dumps(report,indent=2), encoding='utf-8')
print('Core inspection saved',report['vertices'],report['polygons'])
