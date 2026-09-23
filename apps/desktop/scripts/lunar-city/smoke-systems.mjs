import { chromium } from '@playwright/test'
import { writeFile } from 'node:fs/promises'
import assert from 'node:assert/strict'
const url=process.argv[2]??'http://127.0.0.1:5178/lunar-city-review.html'
const evidence=process.argv[3]??'docs/lunar-city/game/evidence/systems-viewer.json'
const browser=await chromium.launch({headless:true})
try {
 const page=await browser.newPage({viewport:{width:1600,height:1100}}),errors=[],external=[]
 page.on('pageerror',e=>errors.push(e.message))
 page.on('request',r=>{if(!r.url().startsWith(new URL(url).origin)&&!r.url().startsWith('blob:')&&!r.url().startsWith('data:'))external.push(r.url())})
 await page.goto(url)
 await page.waitForFunction(()=>document.documentElement.dataset.worldReady==='true',null,{timeout:90000})
 await page.getByRole('button',{name:'Local systems demo',exact:true}).click()
 const attention=page.getByRole('region',{name:'City attention'}),life=page.getByRole('region',{name:'Worker life'})
 await page.getByRole('region',{name:'Project sites'}).getByText('2 observed projects · 2 assigned sites · 0 without a physical site',{exact:true}).waitFor()
 await page.waitForTimeout(5500)
 await page.getByRole('button',{name:'Greet a nearby worker',exact:true}).click()
 await page.waitForFunction(()=>document.querySelector('[aria-label="Worker life"]').textContent.includes('greeting'),null,{timeout:12000})
 const greeting=await life.innerText()
 let actualGestureMotion
 if(new URL(url).port==='5178') {
  const sample=()=>page.evaluate(async()=>{
   const {loadBabylonModules}=await import('/src/app/lunar-city/world/create-world.ts')
   const {Engine}=await loadBabylonModules()
   return Engine.Instances[0].scenes[0].animationGroups.filter(g=>g.isPlaying && /^lunar-city:worker:(talk|listen)$/.test(g.name)).map(g=>({name:g.name,values:g.targetedAnimations.flatMap(t=>{const n=t.target;n.computeWorldMatrix?.(true);return Array.from(n.getWorldMatrix?.().m??[])})}))
  })
  const before=await sample();await page.waitForTimeout(250);const after=await sample()
  assert.ok(before.length>=2,'Greeting must play two actual imported clips')
  actualGestureMotion=after.some((group,i)=>group.values.some((v,j)=>Math.abs(v-(before[i]?.values[j]??v))>1e-7))
  assert.ok(actualGestureMotion,'Imported greeting targets must actually move')
 }
 await page.selectOption('[aria-label="Demo observation"]','blocked')
 await attention.getByText(/· open/).first().waitFor()
 await attention.getByRole('button',{name:/^Acknowledge/}).first().click()
 await attention.getByText(/· acknowledged/).first().waitFor()
 await page.selectOption('[aria-label="Demo observation"]','disconnected')
 assert.match(await attention.innerText(),/acknowledged/)
 await page.selectOption('[aria-label="Demo observation"]','running')
 await attention.getByLabel('Show resolved history').check()
 await attention.getByText(/· resolved/).first().waitFor()
 await page.selectOption('[aria-label="Demo observation"]','blocked')
 await attention.getByText(/· reopened/).first().waitFor()
 const alerts=await attention.innerText()
 await page.getByLabel('Include an observed child worker').check()
 await page.selectOption('[aria-label="Demo observation"]','orchestration')
 await page.getByRole('region',{name:'Observed delegation links'}).waitFor({timeout:12000})
 const delegation=await page.getByRole('region',{name:'Observed delegation links'}).innerText()
 await page.locator('#motion').click()
 await page.getByRole('region',{name:'Observed delegation links'}).waitFor({state:'hidden'})
 await page.screenshot({path:evidence.replace(/\.json$/,'.png')})
 assert.deepEqual(errors,[]);assert.deepEqual(external,[])
 assert.equal(await page.getByRole('button',{name:'Load profile SOUL',exact:true}).isDisabled(),true)
 // Corrupt a declared asset in transit: runtime must reject bytes before Babylon import.
 const bad=await browser.newPage()
 await bad.route('**/models/*.glb',async route=>{const response=await route.fetch(),body=await response.body();body[body.length-1]^=1;await route.fulfill({response,body})})
 await bad.goto(url)
 await bad.waitForFunction(()=>document.querySelector('#loading')?.textContent?.includes('digest mismatch'),null,{timeout:90000})
 const tamper=await bad.locator('#loading').innerText()
 assert.notEqual(await bad.getAttribute('html','data-world-ready'),'true')
 // Navigation integrity must not silently degrade to the declared-link fallback.
 const badNavigation=await browser.newPage()
 await badNavigation.route('**/models/navigation*.glb',async route=>{const response=await route.fetch(),body=await response.body();body[body.length-1]^=1;await route.fulfill({response,body})})
 await badNavigation.goto(url)
 await badNavigation.waitForFunction(()=>document.querySelector('#loading')?.textContent?.includes('digest mismatch'),null,{timeout:90000})
 const navigationTamper=await badNavigation.locator('#loading').innerText()
 assert.notEqual(await badNavigation.getAttribute('html','data-world-ready'),'true')
 await badNavigation.unroute('**/models/navigation*.glb')
 await badNavigation.locator('#retry').click()
 await badNavigation.waitForFunction(()=>document.documentElement.dataset.worldReady==='true',null,{timeout:90000})
 await writeFile(evidence,JSON.stringify({capturedAt:new Date().toISOString(),url,greeting,actualGestureMotion,alerts,delegation,errors,externalRequests:external,tamperRejected:tamper,navigationTamperRejected:navigationTamper,recoveryAfterTamper:true,scope:'Actual local WebGL viewer. Demo observations only; no gateway/API writes.'},null,2))
 console.log({ready:true,greeting:true,alertLifecycle:true,stalePreservesIncident:true,projectSites:2,delegation:true,reducedMotion:true,tamperRejected:true,navigationTamperRejected:true,recoveryAfterTamper:true,errors})
}finally{await browser.close()}
