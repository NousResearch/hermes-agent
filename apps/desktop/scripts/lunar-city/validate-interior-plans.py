"""Validate authored plan circulation independently of Blender exports."""
import json
from pathlib import Path
OUT=Path(__file__).resolve().parents[2]/'public/lunar-city/interior-plans-v1'
data=json.loads((OUT/'plans.json').read_text())
checked=0
for plan in data['plans']:
 for solid in plan['solids']:
  assert min(solid['size'])>0,(plan['id'],solid['id'],'nonpositive solid')
  for void in plan['reservedVoids']:
   if solid['kind']!='floor' or solid['center'][1]-solid['size'][1]/2>=void.get('minimumBridgeUnderside',float('inf')):continue
   x,z,w,d=void['rect'];cx,_,cz=solid['center'];sx,_,sz=solid['size']
   assert cx+sx/2<=x+1e-8 or cx-sx/2>=x+w-1e-8 or cz+sz/2<=z+1e-8 or cz-sz/2>=z+d-1e-8,(plan['id'],solid['id'],'floor closes reserved void')
 for route in plan['routes']:
  for a,b in zip(route,route[1:]):
   checked+=1
   for solid in plan['solids']:
    if solid['kind'] not in ['wall','desk','bench','planter']:continue
    lo,hi=0,1
    for axis in (0,2):
     half=solid['size'][axis]/2+.35;u=a[axis]-solid['center'][axis];v=b[axis]-a[axis]
     if abs(v)<1e-10:
      if abs(u)>half:hi=-1;break
     else:
      aa=(-half-u)/v;bb=(half-u)/v;lo=max(lo,min(aa,bb));hi=min(hi,max(aa,bb))
    assert lo>hi,(plan['id'],route,solid['id'],'blocked route')
 for door in plan['doors']:
  assert door['width']>=1.2 and door['height']>=2.2
receipt=dict(stage='data validation only; Blender shell alignment pending',plans=len(data['plans']),routeSegments=checked,actorRadius=.35,doors='at least1.2m by2.2m',reservedWaterVoids='clear of all floor rectangles')
(OUT/'plan-validation.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
