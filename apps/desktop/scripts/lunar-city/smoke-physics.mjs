import { chromium } from '@playwright/test'
import { writeFile } from 'node:fs/promises'
import { PEDESTRIAN_ROUTES } from './settlement-layout.mjs'
const browser = await chromium.launch({ headless:true })
try {
 const page = await browser.newPage({viewport:{width:1440,height:1000}})
 page.on('console',message=>{if(message.text().startsWith('PHYSICS'))console.log(message.text())})
 const errors=[];page.on('pageerror',error=>errors.push(error.message))
 await page.goto('http://127.0.0.1:5178/lunar-city-review.html')
 await page.waitForFunction(()=>document.documentElement.dataset.worldReady==='true'||!document.querySelector('#retry').hidden,null,{timeout:90000})
 if(await page.locator('#retry').isVisible())throw Error(await page.locator('body').innerText())
 const result=await page.evaluate(async routes=>{
  const {loadBabylonModules,createLunarCityWorld}=await import('/src/app/lunar-city/world/create-world.ts')
  const {loadWorldManifest}=await import('/src/app/lunar-city/manifest.ts')
  const {createRouteNavigationQuery}=await import('/src/app/lunar-city/world/world-navigation.ts')
  const {createNavigationController}=await import('/src/app/lunar-city/world/navigation.ts')
  const modules=await loadBabylonModules(),scene=modules.Engine.Instances[0].scenes[0]
  const url='/lunar-city/v2-review/world-manifest.v2.json',manifest=await loadWorldManifest(url)
  const query=await createRouteNavigationQuery(manifest.navigation,modules,scene,uri=>new URL('/lunar-city/v2-review/'+uri,location.href).href)
  const vec=p=>({x:p[0],y:p[1],z:p[2]}),distance=(a,b)=>Math.hypot(a.x-b.x,a.y-b.y,a.z-b.z)
  const checks=[]
  for(const [index,route]of routes.entries()){
   const start=vec(route.points[0]),end=vec(route.points.at(-1));start.x+=.02
   const entity={key:`physics:route:${index}`,position:start,animation:'idle'}
   const controller=createNavigationController({destinations:{review:end},query,workerClips:new Set(['walk','idle'])})
   if(!controller.move(entity,'review'))throw Error(`Unsafe authored route ${route.from}->${route.to}`)
   let ticks=0,maxStep=0
   while(controller.isMoving(entity.key)&&ticks++<3000){const before={...entity.position};controller.tick(250);maxStep=Math.max(maxStep,distance(before,entity.position));if(!query.resolvePosition(entity.position))throw Error('Actor lost floor/collider safety')}
   if(!controller.didArrive(entity.key)||distance(entity.position,end)>.02||maxStep>.301)throw Error(`Route did not physically arrive ${route.from}->${route.to}`)
   checks.push({from:route.from,to:route.to,ticks,maxStep,arrivalError:distance(entity.position,end)})
  }
  console.log('PHYSICS all routes passed')
  const blocked=manifest.navigation.colliders.map(collider=>({id:collider.id,kind:collider.kind,blocked:query.resolvePosition({x:collider.center.x,y:2.2,z:collider.center.z})===undefined}))
  if(blocked.some(row=>!row.blocked))throw Error('Collider permits actor at solid center')
  if(query.resolvePosition({x:100000,y:-20,z:100000}))throw Error('Offmesh point accepted')
  const {createArrivalSlots}=await import('/src/app/lunar-city/world/arrival-slots.ts')
  const slotTarget=createArrivalSlots(manifest).target('session:connection=physics:profile=worker:session=permanence','lab',manifest.destinations.lab)
  const slotPath=query.computePath(vec(routes[0].points[0]),slotTarget)
  console.log('PHYSICS slot '+JSON.stringify({slotTarget,pathEnd:slotPath?.at(-1),ground:query.resolvePosition(slotTarget),connector:slotPath&&query.canTraverse(slotPath.at(-1),slotTarget)}))
  query.dispose()
  // Use actual imported worker geometry and the normal world lifecycle for permanence.
  const canvas=document.createElement('canvas');canvas.width=800;canvas.height=600;canvas.style.cssText="position:fixed;inset:0;width:800px;height:600px;z-index:9999";document.body.append(canvas)
  let world=await createLunarCityWorld(canvas,manifest,()=>{},undefined,url)
  const first=routes[0],key='session:connection=physics:profile=worker:session=permanence'
  const origin=vec(first.points[0]);const target=first.to==='research-lab'?'lab':first.to
  const entity={animation:'idle',authority:'authoritative',destination:target,identity:{kind:'session',connectionId:'physics',profile:'worker',sessionId:'permanence'},key,observedAt:1,position:origin}
  const snapshot={entities:new Map([[key,entity]]),observedAt:1,revision:1,sources:[]}
  world.applySnapshot(snapshot)
  console.log('PHYSICS initial actor '+JSON.stringify(world.getWorkerPresentation(key)))
  for(let wait=0;wait<150 && (world.getPerfSnapshot?.().renderFrames??0)<3;wait++) await new Promise(resolve=>setTimeout(resolve,100))
  const before=world.getWorkerPresentation(key)
  if(!before||distance(before.position,origin)<.01)throw Error('Actual worker never started walking: '+JSON.stringify({before,origin,target,metrics:world.getPerfSnapshot?.()}))
  world.applySnapshot({...snapshot,revision:2,observedAt:2})
  const afterPoll=world.getWorkerPresentation(key)
  if(distance(before.position,afterPoll.position)>1e-6)throw Error('Polling reset actual worker position')
  world.setReducedMotion(true)
  const paused=world.getWorkerPresentation(key)
  await new Promise(resolve=>setTimeout(resolve,150))
  if(distance(paused.position,world.getWorkerPresentation(key).position)>1e-6)throw Error('Reduced motion moved actor')
  world.destroy()
  world=await createLunarCityWorld(canvas,manifest,()=>{},undefined,url)
  world.applySnapshot(snapshot)
  const restored=world.getWorkerPresentation(key)
  if(!restored||distance(paused.position,restored.position)>1e-6)throw Error('World remount reset actor')
  world.destroy();canvas.remove()
  return {routes:checks,colliders:blocked,permanence:{before,afterPoll,paused,restored},scope:'Actual Recast routes through movement controller; collider center/offmesh holds; real imported worker polling/pause/world-remount identity. In-memory renderer scope, not browser reload persistence.'}
 },PEDESTRIAN_ROUTES)
 if(errors.length)throw Error(errors.join('\n'))
 await writeFile('docs/lunar-city/game/evidence/physics-runtime.json',JSON.stringify({...result,errors,capturedAt:new Date().toISOString()},null,2))
 console.log({pass:true,routes:result.routes.length,colliders:result.colliders.length,permanence:true})
}finally{await browser.close()}
