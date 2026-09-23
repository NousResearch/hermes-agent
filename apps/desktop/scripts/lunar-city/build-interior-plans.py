"""Author basic independent interior studies from the current placed building bounds."""
import hashlib, json, html, math
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'public/lunar-city/interior-plans-v1'
OUT.mkdir(parents=True, exist_ok=True)
source = ROOT / 'public/lunar-city/v2-review/world-manifest.v2.json'
manifest = json.loads(source.read_text())
roles = {
 'library':['Book intake','Reading room','Stacks','Librarian desk'],
 'research-lab':['Sample reception','Analysis benches','Instrument lab','Observation control'],
 'depot':['Receiving','Dispatch','Parts storage','Packing benches'],
 'review-office':['Review reception','Peer review room','Evidence records','Decision meeting'],
 'triage':['Incident intake','Diagnostics','Repair benches','Recovery station'],
 'council':['Public reception','Council chamber','Steward office','Planning records'],
 'arts-studio':['Materials store','Painting studio','Digital workbench','Critique room'],
 'release-gatehouse':['Release intake','Validation benches','Staging store','Release control'],
 'archive':['Acquisition desk','Catalog room','Protected records','Memory curator'],
 'publishing':['Editorial intake','Editing desks','Print preparation','Publishing desk'],
 'revenue':['Visitor reception','Accounts workspace','Secure records','Revenue office'],
}
measurements={row['id']:row for row in json.loads((OUT/'actor-clearances.json').read_text())['actors']}
leader_buildings={'library':'owl','research-lab':'fox','archive':'elephant','arts-studio':'cat','revenue':'capybara','council':'lion','engineering-workshop':'beaver','publishing':'monkey'}
plans=[]
def box(id,kind,x,y,z,w,h,d): return dict(id=id,kind=kind,center=[x,y,z],size=[w,h,d])
for model in manifest['models']:
 id=model['id']
 if id in ['terrain','workers','leaders']: continue
 bounds=model['bounds']; scale=model['transform']['scale']
 extent=[(bounds['max'][i]-bounds['min'][i])*scale[i] for i in range(3)]
 w,d=round(extent[0]*.78,2),round(extent[2]*.78,2)
 actor=measurements.get(leader_buildings.get(id,''));clear=max(1.5,actor['clearWidth'] if actor else 1.5);radius=actor['radiusMetres'] if actor else .35
 plan=dict(id=id,title=id.replace('-',' ').title(),units='metres',axes='Y up; +Z front',sourceModelUri=model['uri'],sourceModelSha256=model['statistics']['sha256'],placedExteriorBounds=extent,exteriorTransform=model['transform'],fitStatus='authored_plan_within_bounds_not_surveyed_shell_fit',ceilingClearance=3.0,doorClearWidth=clear,doorClearHeight=2.6,circulationClearWidth=clear,leaderEnvelope=actor,footprint=[w,d],rooms=[],solids=[],doors=[],routes=[],reservedVoids=[],ramps=[],expansion=[],notes=[])
 def add(*args): plan['solids'].append(box(*args))
 if id in ['garden','bus']:
  plan['enclosure']='open_air';plan['ceilingClearance']=None
  plan['notes'].append('Open-air layout; no invented enclosed room or 3 m interior beneath the bus asset which is only 3 m tall overall.')
  add('court-floor','floor',0,-.075,0,w,.15,d)
  if id=='garden':
   for i,(x,z) in enumerate([(-w*.3,-d*.25),(w*.3,-d*.25),(-w*.3,d*.25),(w*.3,d*.25)]):
    plan['rooms'].append(dict(id=f'bed-{i}',name=['Learning beds','Herb nursery','Seed exchange','Quiet seating'][i],rect=[x-w*.13,z-d*.13,w*.26,d*.26]))
    add(f'bed-{i}','planter',x,.3,z,w*.26,.6,d*.26)
   plan['routes']=[[[0,0,d/2],[0,0,-d/2]],[[-w/2,0,0],[w/2,0,0]]]
  else:
   for side in [-1,1]: add(f'bench-{side}','bench',side*w*.32,.23,0,.5,.46,d*.62)
   plan['rooms']=[dict(id='waiting',name='Open waiting and message-routing bay',rect=[-.75,-d/2,1.5,d])]
   plan['routes']=[[[0,0,d/2],[0,0,-d/2]]]
  plan['expansion']=[dict(id='future-shelter',position=[0,0,-d/2],note='Future canopy/service pod only after exterior and clearance review')]
 elif id=='engineering-workshop':
  plan['enclosure']='bank_rooms_with_open_water_channel';w=round(extent[0]-.8,2);d=round(extent[2]-.8,2);plan['footprint']=[w,d]
  channel=6.; plan['reservedVoids']=[dict(id='river-channel',rect=[-channel/2,-d/2,channel,d],floorForbidden=True)]
  bank=(w-channel)/2
  for side in [-1,1]:
   x=side*(channel/2+bank/2)
   add(f'bank-floor-{side}','floor',x,-.075,0,bank,.15,d)
   plan['rooms'].append(dict(id=f'bank-{side}',name='Mechanical workshop' if side<0 else 'Waterworks control',rect=[x-bank/2,-d/2,bank,d]))
   for zz in [-d/2,d/2]:
    for shift in [-1,1]: add(f'end-{side}-{zz}-{shift}','wall',x+shift*(bank/4+clear/4),1.5,zz,(bank-clear)/2,3,.15)
    plan['doors'].append(dict(id=f'bank-door-{side}-{zz}',center=[x,0,zz],width=clear,height=2.6,axis='x'))
   for xx in [x-bank/2,x+bank/2]: add(f'bank-wall-{side}-{xx}','wall',xx,1.5,0,.15,3,d)
   add(f'bench-{side}','desk',x+side*(bank/2-.55),.4,-d*.22,.8,.8,2.4)
   plan['routes'].append([[x,0,d/2],[x,0,-d/2]])
  plan['expansion']=[dict(id='future-upper-gallery',position=[0,3.4,0],note='Separate future bridge above water; no deck or floor currently closes channel')]
  plan['notes'].append('Six-metre-wide water channel retained beneath the raised crossing; both bank rooms have front/rear access.')
 else:
  plan['enclosure']='single_storey_cutaway'
  # 0.15m walls; corridor width is clear between inner wall faces.
  wall=.15;inner=clear/2+wall/2;outer=w/2-wall/2
  add('floor','floor',0,-.075,0,w,.15,d)
  add('rear-wall','wall',0,1.5,-d/2,w,3,wall)
  for side in [-1,1]:
   add(f'outer-{side}','wall',side*w/2,1.5,0,wall,3,d)
   add(f'entry-jamb-{side}','wall',side*(w/4+clear/4),1.5,d/2,(w-clear)/2,3,wall)
  add('entry-header','header',0,2.8,d/2,clear,.4,wall)
  plan['doors'].append(dict(id='front-entry',center=[0,0,d/2],width=clear,height=2.6,axis='x'))
  for side in [-1,1]:
   roomw=outer-inner-wall
   centerx=side*(inner+wall/2+roomw/2)
   add(f'room-divider-{side}','wall',centerx,1.5,0,roomw,3,wall)
   for row in [-1,1]:
    centerz=row*d/4
    index=(0 if row>0 else 2)+(0 if side<0 else 1);name=roles[id][index]
    plan['rooms'].append(dict(id=f'room-{index}',name=name,rect=[inner+wall/2 if side>0 else -outer, centerz-d/4+wall/2, outer-inner-wall/2, d/2-wall]))
    doorz=centerz
    for segment,lo,hi in [('a',centerz-d/4,doorz-clear/2),('b',doorz+clear/2,centerz+d/4)]:
     add(f'corridor-{side}-{row}-{segment}','wall',side*inner,1.5,(lo+hi)/2,wall,3,hi-lo)
    add(f'door-header-{side}-{row}','header',side*inner,2.8,doorz,wall,.4,clear)
    plan['doors'].append(dict(id=f'room-door-{index}',center=[side*inner,0,doorz],width=clear,height=2.6,axis='z'))
    add(f'desk-{index}','desk',side*(w/2-.75),.4,centerz,.85,.8,min(1.8,d/2-.9))
    plan['routes'].append([[0,0,doorz],[side*((inner+wall/2)+(w/2-1.175))/2,0,doorz]])
  plan['routes'].append([[0,0,d/2],[0,0,-d/2+radius+.2]])
  add('ceiling-reference','ceiling',0,3.075,0,w,.15,d)
  plan['expansion']=[dict(id='rear-service-growth',position=[0,0,-d/2],note='Future room/vertical access reserve; fit against actual shell before opening rear wall')]
  if id=='archive': plan['notes'].append('Ground-level trunk pavilion prototype only; canopy pods remain future separate rooms. Rectangular bbox fit does not prove fit within organic trunk.')
  if id=='research-lab': plan['notes'].append('Instrument/control floor only. Telescope dome access and circular-shell wall fitting are future work.')
  if id=='council': plan['notes'].append('Ground-level public/work floor only; stairs, chamber proportions and upper levels are future design.')
 # Anchors are placed on declared circulation, never inside furniture.
 plan['anchors']={}
 for i,route in enumerate(plan['routes']):
  plan['anchors'][f'workspace-{i}']=route[-1]
 main=plan['routes'][-1] if id not in ['engineering-workshop','garden','bus'] else plan['routes'][0]
 plan['anchors'].update(entry=main[0],home=main[len(main)//2],work=plan['routes'][0][-1],rest=main[-1],jobPickup=plan['routes'][0][-1])
 # Portal approaches use the actual exported building's full depth, with room for
 # an actor outside its former OBB. The collar floor joins plan to that approach.
 entries=[main[0]] if id!='engineering-workshop' else [r[0] for r in plan['routes']]
 plan['runtimePortalStatus']='authored_approach'
 plan['portals']=[]
 for i,entry in enumerate(entries):
  target=manifest['destinations'].get({'research-lab':'lab','review-office':'review'}.get(id,id))
  yaw=model['transform']['rotation'][1];pos=model['transform']['position']
  if target and id!='engineering-workshop':
   dx,dz=target[0]-pos[0],target[2]-pos[2]
   approach=[dx*math.cos(yaw)-dz*math.sin(yaw),0,dx*math.sin(yaw)+dz*math.cos(yaw)]
  else: approach=[entry[0],0,max(entry[2],extent[2]/2)+.85]
  plan['portals'].append(dict(id=f'entry-{i}',entry=entry,approach=approach,width=clear))
  length=approach[2]-entry[2]
  add(f'portal-floor-{i}','floor',entry[0],-.075,(approach[2]+entry[2])/2,clear,.15,length)
 if id!='engineering-workshop':
  plan['anchors']['home']=[0,0,0]
 if id=='engineering-workshop':
  rise=.8
  for solid in plan['solids']: solid['center'][1]+=rise
  for route in plan['routes']:
   for point in route: point[1]+=rise
  for door in plan['doors']: door['center'][1]+=rise
  plan['reservedVoids'][0]['minimumBridgeUnderside']=.6
  bridgez=d/2-3.0
  bankx=channel/2+bank/2
  plan['solids']=[solid for solid in plan['solids'] if not (solid['kind']=='wall' and abs(abs(solid['center'][0])-channel/2)<.01)]
  for side in [-1,1]:
   for label,lo,hi in [('rear',-d/2,bridgez-clear/2),('front',bridgez+clear/2,d/2)]:
    add(f'bridge-bank-wall-{side}-{label}','wall',side*channel/2,rise+1.5,(lo+hi)/2,.15,3,hi-lo)
   add(f'bridge-door-header-{side}','header',side*channel/2,rise+2.8,bridgez,.15,.4,clear)
   plan['doors'].append(dict(id=f'bridge-bank-door-{side}',center=[side*channel/2,rise,bridgez],width=clear,height=2.6,axis='z'))
  add('elevated-channel-catwalk','floor',0,rise-.075,bridgez,channel,.15,clear)
  plan['routes'].append([[-bankx,rise,bridgez],[bankx,rise,bridgez]])
  target=manifest['destinations'][id];yaw=model['transform']['rotation'][1];pos=model['transform']['position']
  dx,dz=target[0]-pos[0],target[2]-pos[2]
  approach=[dx*math.cos(yaw)-dz*math.sin(yaw),0,dx*math.sin(yaw)+dz*math.cos(yaw)]
  entry=[0,rise,bridgez+clear/2]
  plan['ramps']=[dict(id='waterworks-entry-ramp',start=approach,end=entry,width=clear,thickness=.12)]
  plan['portals']=[dict(id='raised-channel-entry',entry=entry,approach=approach,width=clear)]
  plan['routes'].append([approach,entry])
  plan['routes'].append([entry,[0,rise,bridgez]])
  plan['anchors'].update(entry=entry,home=[-bankx,rise,bridgez],rest=[bankx,rise,bridgez],work=[-bankx,rise,-d*.1],jobPickup=[-bankx,rise,bridgez])
  plan['notes'].append('Authored 0.8 m raised bank floors and 1.5 m catwalk; 0.65 m underside above local datum keeps the water channel open below. Entry ramp joins the existing outdoor approach; river/road edge requires visible alignment review.')
 if id in leader_buildings and id!='engineering-workshop':
  room_index={'library':3,'research-lab':3,'archive':3,'arts-studio':1,'revenue':3,'council':2,'publishing':3}[id]
  route_index={2:0,0:1,3:2,1:3}[room_index]
  plan['anchors']['leaderWork']=plan['routes'][route_index][-1]
 elif id=='engineering-workshop':plan['anchors']['leaderWork']=plan['anchors']['work']
 # Local +Z rotates consistently with the source model's Babylon yaw.
 yaw=plan['exteriorTransform']['rotation'][1]; pos=plan['exteriorTransform']['position']
 def world(v): return [pos[0]+v[0]*math.cos(yaw)+v[2]*math.sin(yaw),pos[1]+v[1],pos[2]-v[0]*math.sin(yaw)+v[2]*math.cos(yaw)]
 plan['worldAnchors']={k:world(v) for k,v in plan['anchors'].items()}
 plan['notes'].append('New authored interior design based on role and current scale, not recovered original floorplans. Exterior entrances and negative spaces still require visible Blender alignment.')
 plans.append(plan)
result=dict(version=1,stage='basic_authored_cutaway_review',worldManifestSha256=hashlib.sha256(source.read_bytes()).hexdigest(),plans=plans)
(OUT/'plans.json').write_text(json.dumps(result,indent=2)+'\n')
# A simple readable browser companion; SVG is generated plan geometry, not image editing.
svgs=[]
colors={'wall':'#455667','header':'#455667','desk':'#bb8359','floor':'#edf0ed','planter':'#6ba486','bench':'#bb8359'}
for p in plans:
 w,d=p['footprint'];items=[]
 for v in p['reservedVoids']:
  x,z,ww,dd=v['rect'];items.append(f'<rect x="{x}" y="{-z-dd}" width="{ww}" height="{dd}" fill="#9ed9df"/><text x="0" y="0" text-anchor="middle" font-size=".42">OPEN WATER</text>')
 for s in p['solids']:
  if s['kind'] in ['ceiling','header']:continue
  x,y,z=s['center'];sx,sy,sz=s['size'];items.append(f'<rect x="{x-sx/2}" y="{-z-sz/2}" width="{sx}" height="{sz}" fill="{colors[s["kind"]]}"/>')
 for ramp in p.get('ramps',[]):
  a,b=ramp['start'],ramp['end'];items.append(f'<line x1="{a[0]}" y1="{-a[2]}" x2="{b[0]}" y2="{-b[2]}" stroke="#c1d0c8" stroke-width="{ramp["width"]}"/>')
 for r in p['rooms']:
  x,z,ww,dd=r['rect'];items.append(f'<text x="{x+ww/2}" y="{-z-dd/2}" text-anchor="middle" font-size=".28">{html.escape(r["name"])}</text>')
 for route in p['routes']: items.append('<polyline points="'+' '.join(f'{x},{-z}' for x,y,z in route)+'" fill="none" stroke="#16816b" stroke-width=".05" stroke-dasharray=".15 .1"/>')
 svg=f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{-w/2-.7} {-d/2-.7} {w+1.4} {d+1.4}">'+''.join(items)+'</svg>'
 (OUT/f'{p["id"]}.svg').write_text(svg)
 svgs.append(f'<article><h2>{html.escape(p["title"])}</h2>{svg}<p>{w} × {d} m · '+('Open air' if p['ceilingClearance'] is None else '3 m clear ceiling')+'</p><p>'+html.escape(' '.join(p['notes']))+'</p></article>')
(OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Lunar City basic floorplans</title><style>body{font:15px system-ui;background:#e6eeed;color:#243a3a;margin:32px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:24px}article{background:white;border-radius:16px;padding:24px}svg{width:100%;height:360px}p{line-height:1.5}h1{font-size:30px}</style><h1>Lunar City · basic interior studies</h1><p>Fourteen separate plans. Newly authored, metre-scaled prototypes; exterior shell/door alignment still needs Blender review. Dashed lines are circulation. Walls are shown cut away. No live game navigation changes.</p><main>'+''.join(svgs)+'</main>')
print(f'Wrote {len(plans)} floorplans to {OUT}')
