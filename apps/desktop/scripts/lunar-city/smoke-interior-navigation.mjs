import { chromium } from '@playwright/test'
import { writeFile } from 'node:fs/promises'
const browser=await chromium.launch({headless:true})
try {
 const page=await browser.newPage({viewport:{width:1600,height:1000}})
 await page.goto('http://127.0.0.1:5178/lunar-city-review.html')
 await page.waitForFunction(()=>document.documentElement.dataset.worldReady==='true',null,{timeout:120000})
 const result=await page.evaluate(async()=>{
  const {loadBabylonModules,createLunarCityWorld}=await import('/src/app/lunar-city/world/create-world.ts')
  const {loadWorldManifest}=await import('/src/app/lunar-city/manifest.ts')
  const {createRouteNavigationQuery}=await import('/src/app/lunar-city/world/world-navigation.ts')
  const {eligibleInteriorPlans}=await import('/src/app/lunar-city/world/interior-plan-data.ts')
  const {interiorWorldPoint}=await import('/src/app/lunar-city/world/interior-navigation.ts')
  const modules=await loadBabylonModules(),scene=modules.Engine.Instances[0].scenes[0]
  const manifest=await loadWorldManifest('/lunar-city/v2-review/world-manifest.v2.json'),plans=eligibleInteriorPlans(manifest)
  const query=await createRouteNavigationQuery(manifest.navigation,modules,scene,uri=>new URL(`/lunar-city/v2-review/${uri}`,location.href).href,plans)
  const {createNavigationController}=await import('/src/app/lunar-city/world/navigation.ts')
  const entries=[]
  for(const plan of plans){
   const from=interiorWorldPoint(plan,plan.portals[0].approach),p=plan.worldAnchors.jobPickup,to={x:p[0],y:p[1],z:p[2]}
   const path=query.computePath(from,to)
   if(!path?.length)throw new Error(`${plan.id}: no pickup route; fromground=${JSON.stringify(query.resolvePosition(from))} toground=${JSON.stringify(query.resolvePosition(to))}`)
   for(let i=1;i<path.length;i++){if(!query.canTraverse(path[i-1],path[i]))throw new Error(`${plan.id}: unsafe segment ${i}`)}
   const wall=plan.solids.find(s=>s.kind==='wall')
   if(wall&&query.resolvePosition(interiorWorldPoint(plan,[wall.center[0],wall.center[1]-wall.size[1]/2,wall.center[2]])))throw new Error(`${plan.id}: wall not solid`)
   const actor={key:`interior:${plan.id}`,position:{...from},animation:'idle'}
   const controller=createNavigationController({destinations:{review:to},query,workerClips:new Set(['walk','idle'])})
   if(!controller.move(actor,'review'))throw new Error(`${plan.id}: controller rejected path`)
   let ticks=0
   while(controller.isMoving(actor.key)&&ticks++<600){controller.tick(100);if(!query.resolvePosition(actor.position))throw new Error(`${plan.id}: lost floor`)}
   if(!controller.didArrive(actor.key))throw new Error(`${plan.id}: failed physical arrival`)
   entries.push({id:plan.id,from,to,path,ticks,wallRejected:!!wall})
  }
  // An assigned worker from another district must reach the room through the outdoor network.
  const start=manifest.destinations.depot,cat=plans.find(p=>p.id==='arts-studio').worldAnchors.jobPickup
  const cross=query.computePath(start,{x:cat[0],y:cat[1],z:cat[2]})
  if(!cross?.length||cross.some((p,i)=>i&&!query.canTraverse(cross[i-1],p)))throw new Error('depot to cat studio route unsafe')
  query.dispose?.()
  const canvas=document.createElement('canvas');canvas.width=800;canvas.height=600;canvas.style.cssText='position:fixed;inset:0;width:800px;height:600px;z-index:9999';document.body.append(canvas)
  const world=await createLunarCityWorld(canvas,manifest,()=>{},undefined,'/lunar-city/v2-review/world-manifest.v2.json')
  const plan=plans.find(p=>p.id==='arts-studio'),origin=interiorWorldPoint(plan,plan.portals[0].entry),key='session:connection=interior:profile=worker:session=pickup'
  world.setInteriorBuilding('arts-studio')
  const entity={animation:'idle',authority:'authoritative',destination:'arts-studio',identity:{kind:'session',connectionId:'interior',profile:'worker',sessionId:'pickup'},key,observedAt:1,position:origin}
  const snapshot={entities:new Map([[key,entity]]),observedAt:1,revision:1,sources:[]}
  world.applySnapshot(snapshot)
  const expected=plan.worldAnchors.jobPickup
  let arrived=false,last
  for(let count=0;count<400;count++){
   await new Promise(resolve=>setTimeout(resolve,100))
   last=world.getWorkerPresentation(key)
   if(last&&!last.moving&&Math.hypot(last.position.x-expected[0],last.position.y-expected[1],last.position.z-expected[2])<.02){arrived=true;break}
  }
  if(!arrived)throw new Error('Actual worker did not reach assigned indoor pickup '+JSON.stringify({last,expected}))
  world.applySnapshot({...snapshot,revision:2,observedAt:2})
  const afterPoll=world.getWorkerPresentation(key)
  world.setReducedMotion(true)
  const actualWorker={origin,arrived:last,afterPoll}
  world.destroy();canvas.remove()
  return {plans:entries.length,entries,actualWorker,crossDistrictPath:cross,navSha256:manifest.navigation.sha256}
 })
 await writeFile('docs/lunar-city/game/evidence/interior-navigation-runtime.json',JSON.stringify({capturedAt:new Date().toISOString(),...result},null,2)+'\n')
 console.log(JSON.stringify({plans:result.plans,crossDistrictSegments:result.crossDistrictPath.length-1}))
} finally {await browser.close()}
