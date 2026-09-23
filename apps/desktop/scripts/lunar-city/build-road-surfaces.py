"""Union road corridors before triangulation. Requires shapely==2.1.2."""
import json,subprocess,hashlib
from pathlib import Path
from shapely import constrained_delaunay_triangles
from shapely.geometry import LineString
from shapely.ops import unary_union
root=Path(__file__).resolve().parent
raw=subprocess.check_output(['node','--input-type=module','-e',"import {PEDESTRIAN_ROUTES,ROAD_WIDTH} from './settlement-layout.mjs'; console.log(JSON.stringify({routes:PEDESTRIAN_ROUTES,width:ROAD_WIDTH}))"],cwd=root)
data=json.loads(raw)
lines=[LineString([(p[0],p[2]) for p in r['points']]) for r in data['routes']]
rows={}
for name,extra,y in [('road',0,2.20),('paving',.8,2.18),('gravel',1.25,2.035)]:
 shape=unary_union([line.buffer(data['width']/2+extra,quad_segs=6,join_style='round') for line in lines])
 vertices=[];indices=[]
 for triangle in constrained_delaunay_triangles(shape).geoms:
  pts=list(triangle.exterior.coords)[:3]
  if (pts[1][0]-pts[0][0])*(pts[2][1]-pts[0][1])-(pts[1][1]-pts[0][1])*(pts[2][0]-pts[0][0])<0:pts.reverse()
  start=len(vertices)//3
  for x,z in pts:vertices.extend([x,y,z])
  indices.extend([start,start+1,start+2])
 rows[name]={'positions':vertices,'indices':indices,'areaMetresSquared':shape.area}
output=root/'road-surfaces.json'
output.write_text(json.dumps({'sourceSha256':hashlib.sha256(raw.strip()).hexdigest(),'generator':'Shapely2.1.2 buffer union constrained triangulation','layers':rows},separators=(',',':'))+'\n')
print('Union road triangles:',{name:len(row['indices'])//3 for name,row in rows.items()})
