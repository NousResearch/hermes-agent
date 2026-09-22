import { chromium } from '@playwright/test'
import assert from 'node:assert/strict'
import { writeFile } from 'node:fs/promises'
const browser=await chromium.launch({headless:true})
try {
 const page=await browser.newPage()
 const initializationStarted=Date.now()
 await page.goto('http://127.0.0.1:5178/lunar-city-review.html')
 await page.waitForFunction(()=>document.documentElement.dataset.worldReady==='true',null,{timeout:120000})
 const initializationMs=Date.now()-initializationStarted
 await page.selectOption('#leaders','lunar-city:leader:cat')
 assert.equal(await page.locator('#leader-life').isEnabled(),true)
 const observe=()=>page.evaluate(async()=>{
  const {loadBabylonModules}=await import('/src/app/lunar-city/world/create-world.ts')
  const modules=await loadBabylonModules(),scene=modules.Engine.Instances[0].scenes[0]
  const root=scene.transformNodes.find(node=>node.name==='review-leader:cat')
  return {position:{x:root.position.x,y:root.position.y,z:root.position.z},walkPlaying:scene.animationGroups.some(group=>group.name==='leader:cat:walk'&&group.isPlaying===true),playing:scene.animationGroups.filter(group=>group.isPlaying).map(group=>group.name)}
 })
 const plans=await (await page.request.get('http://127.0.0.1:5178/lunar-city/interior-plans-v1/plans.json')).json()
 const anchors=plans.plans.find(plan=>plan.id==='arts-studio').worldAnchors
 const distance=(a,b)=>Math.hypot(a.x-b[0],a.y-b[1],a.z-b[2])
 const residents=await page.evaluate(async()=>{
  const {loadBabylonModules}=await import('/src/app/lunar-city/world/create-world.ts')
  const modules=await loadBabylonModules(),scene=modules.Engine.Instances[0].scenes[0]
  return scene.transformNodes.filter(node=>node.name.startsWith('review-leader:')&&!node.name.startsWith('review-leader:worker')).map(node=>({id:node.name.slice('review-leader:'.length),position:{x:node.position.x,y:node.position.y,z:node.position.z}}))
 })
 const homes={owl:'library',fox:'research-lab',elephant:'archive',cat:'arts-studio',capybara:'revenue',lion:'council',beaver:'engineering-workshop',monkey:'publishing'}
 for(const [id,building] of Object.entries(homes)) {
  const actual=residents.find(row=>row.id===id),home=plans.plans.find(plan=>plan.id===building).worldAnchors.home
  assert.ok(actual&&distance(actual.position,home)<.03,JSON.stringify({id,actual,home}))
 }
 const receipt={initializationMs,residents,initial:await observe(),legs:[]}
 await page.evaluate(async()=>{
  const {loadBabylonModules}=await import('/src/app/lunar-city/world/create-world.ts')
  const modules=await loadBabylonModules(),scene=modules.Engine.Instances[0].scenes[0]
  window.__leaderFrames=[]
  scene.onAfterRenderObservable.add(()=>{
   const root=scene.transformNodes.find(node=>node.name==='review-leader:cat')
   if(window.__leaderFrames.length<3000)window.__leaderFrames.push({time:performance.now(),x:root.position.x,z:root.position.z,playing:scene.animationGroups.filter(group=>group.isPlaying).map(group=>group.name)})
  })
 })
 for(const mode of ['work','home']) {
  const firstFrame=await page.evaluate(()=>window.__leaderFrames.length)
  await page.selectOption('#leader-life',mode)
  const target=anchors[mode==='work'?'leaderWork':'home']
  let seenWalk=false,last
  for(let step=0;step<300;step++) {
   await page.waitForTimeout(100)
   last=await observe();seenWalk ||= last.walkPlaying
   if(distance(last.position,target)<.03&&!last.walkPlaying)break
  }
  const frames=await page.evaluate(first=>window.__leaderFrames.slice(first),firstFrame)
  seenWalk ||= frames.some(frame=>frame.playing.includes('leader:cat:walk'))
  receipt.legs.push({mode,target,last,seenWalk,frames});console.log(JSON.stringify({mode,seenWalk,frames:frames.length}))
  assert.ok(distance(last.position,target)<.03,JSON.stringify(receipt))
  assert.equal(seenWalk,true)
 }
 await page.selectOption('#leader-life','idle')
 const before=await observe()
 await page.click('#motion')
 await page.waitForTimeout(500)
 assert.deepEqual((await observe()).position,before.position)
 await page.click('#motion')
 let stroll
 for(let step=0;step<650;step++) {
  await page.waitForTimeout(100)
  stroll=await observe()
  if(stroll.walkPlaying&&distance(stroll.position,Object.values(before.position))>.1)break
 }
 assert.equal(stroll.walkPlaying,true)
 receipt.stroll={before,actual:stroll};receipt.reducedMotionHeld=true
 await page.click('#motion')
 const held=await observe()
 await page.waitForTimeout(400)
 assert.deepEqual((await observe()).position,held.position)
 receipt.holdWhileWalking=true
 receipt.remount=await page.evaluate(async()=>{
  const {createLunarCityWorld}=await import('/src/app/lunar-city/world/create-world.ts')
  const {loadWorldManifest}=await import('/src/app/lunar-city/manifest.ts')
  const manifest=await loadWorldManifest('/lunar-city/v2-review/world-manifest.v2.json')
  const canvas=document.createElement('canvas');canvas.width=300;canvas.height=200;document.body.append(canvas)
  let world
  try {
   world=await createLunarCityWorld(canvas,manifest,()=>{},undefined,'/lunar-city/v2-review/world-manifest.v2.json')
   world.setLeaderLifeMode('cat','work')
   const origin=world.getLeaderLife('cat').position
   world.setReducedMotion(false)
   let traveled=false
   for(let tick=0;tick<150;tick++) {
    await new Promise(resolve=>setTimeout(resolve,100))
    const current=world.getLeaderLife('cat').position
    if(Math.hypot(current.x-origin.x,current.y-origin.y,current.z-origin.z)>.15){traveled=true;break}
   }
   if(!traveled)throw new Error('Remount fixture did not depart initial position '+JSON.stringify(world.getLeaderLife('cat')))
   world.setReducedMotion(true)
   const before=world.getLeaderLife('cat')
   world.destroy()
   world=await createLunarCityWorld(canvas,manifest,()=>{},undefined,'/lunar-city/v2-review/world-manifest.v2.json')
   world.setReducedMotion(true)
   const after=world.getLeaderLife('cat')
   if(JSON.stringify(before.position)!==JSON.stringify(after.position))throw new Error('Leader position lost across remount')
   return {before,after}
  }finally{world?.destroy();canvas.remove()}
 })
 await writeFile('docs/lunar-city/game/evidence/leader-life-runtime.json',JSON.stringify({capturedAt:new Date().toISOString(),...receipt},null,2))
 console.log(JSON.stringify(receipt))
}finally{await browser.close()}
