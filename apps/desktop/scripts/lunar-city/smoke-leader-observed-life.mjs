import { chromium } from '@playwright/test'
import { writeFile } from 'node:fs/promises'
const browser=await chromium.launch({headless:true})
try {
 const page=await browser.newPage()
 await page.goto('http://127.0.0.1:5178/lunar-city-review.html')
 await page.waitForFunction(()=>document.documentElement.dataset.worldReady==='true',null,{timeout:120000})
 await page.click('#motion')
 const result=await page.evaluate(async()=>{
  const {createLunarCityWorld,loadBabylonModules}=await import('/src/app/lunar-city/world/create-world.ts')
  const {loadWorldManifest}=await import('/src/app/lunar-city/manifest.ts')
  const {leaderModelIdForOwner}=await import('/src/app/lunar-city/leader-runtime.ts')
  const manifest=await loadWorldManifest('/lunar-city/v2-review/world-manifest.v2.json')
  const models=manifest.characterAssets.leaders.map(entry=>entry.id)
  const findOwner=connectionId=>{
   for(let index=0;index<200;index++) {
    const owner={connectionId,profile:`fixture-${index}`}
    if(leaderModelIdForOwner(owner,models)==='cat')return owner
   }
   throw new Error('No exact owner maps to cat')
  }
  const owner=findOwner('local-proof-a'),other=findOwner('local-proof-b')
  const profile=value=>({key:`profile:${value.connectionId}:${value.profile}`,identity:{kind:'profile',...value},presentation:{groups:[],metadata:{source:'local-fixture',state:'fresh'},placement:{lodHint:2,overflow:true}},authority:'authoritative',observedAt:1,destination:'garden',animation:'rest'})
  const session=(animation='work',authority='authoritative')=>({key:`session:${owner.connectionId}:${owner.profile}:proof`,identity:{kind:'session',...owner,sessionId:'proof'},authority,observedAt:1,destination:'garden',animation})
  const canvas=document.createElement('canvas');canvas.width=400;canvas.height=300;document.body.append(canvas)
  let world
  const events=[]
  try {
   world=await createLunarCityWorld(canvas,manifest,()=>{},undefined,'/lunar-city/v2-review/world-manifest.v2.json')
   const modules=await loadBabylonModules(),scene=modules.Engine.Instances.at(-1).scenes[0]
   const clipFrames=[]
   scene.onAfterRenderObservable.add(()=>{if(scene.animationGroups.some(group=>group.name==='leader:cat:walk'&&group.isPlaying))clipFrames.push(performance.now())})
   let revision=0
   const publish=(...entities)=>world.applySnapshot({entities:new Map(entities.map(entity=>[entity.key,entity])),revision:++revision,observedAt:revision,sources:[]})
   const check=(label,mode,manualOverride=false)=>{
    const value=world.getLeaderLife('cat');events.push({label,...value})
    if(value.mode!==mode||value.manualOverride!==manualOverride)throw new Error(label+JSON.stringify(value))
    return value
   }
   const initial=world.getLeaderLife('cat').position
   publish(profile(owner),session())
   check('unique observed work','work')
   let moved=false
   for(let step=0;step<150;step++){
    await new Promise(resolve=>setTimeout(resolve,100))
    const current=world.getLeaderLife('cat').position
    if(Math.hypot(current.x-initial.x,current.y-initial.y,current.z-initial.z)>.15){moved=true;break}
   }
   if(!moved||!clipFrames.length)throw new Error('Observed work did not play actual moving walk')
   publish(profile(owner),session('idle'))
   check('observed idle','idle')
   publish(profile(owner),session())
   publish(profile(owner),session('work','stale'))
   const stale=check('stale owner/session hold','unavailable')
   await new Promise(resolve=>setTimeout(resolve,350))
   if(JSON.stringify(stale.position)!==JSON.stringify(world.getLeaderLife('cat').position))throw new Error('Stale moved')
   publish(profile(owner),session())
   publish(profile(owner),profile(other),session())
   check('ambiguous shared model hold','unavailable')
   publish(profile(owner),session())
   world.setLeaderLifeMode('cat','home')
   check('manual preview override','home',true)
   publish(profile(owner),session('idle'))
   check('observations do not replace manual preview','home',true)
   world.setLeaderLifeMode('cat','automatic')
   check('automatic restores last observation','idle')
   world.setReducedMotion(true)
   return {owner,other,modelIds:models,catUri:manifest.reviewLeaderAssets.find(asset=>asset.id==='cat').uri,events,actualWalkFrames:clipFrames.length,scope:'Synthetic immutable snapshots exercise actual world projection and imported cat motion. No backend requests or work started.'}
  }finally{world?.destroy();canvas.remove()}
 })
 await writeFile('docs/lunar-city/game/evidence/leader-observed-life-runtime.json',JSON.stringify({capturedAt:new Date().toISOString(),...result},null,2))
 console.log(JSON.stringify(result))
}finally{await browser.close()}
