import { chromium } from '@playwright/test'
import {writeFile} from 'node:fs/promises'
import {createHash} from 'node:crypto'
const browser=await chromium.launch({headless:true})
try {
 const page=await browser.newPage({viewport:{width:1440,height:1000}}), errors=[], loaded=[]
 page.on('pageerror',e=>errors.push(e.message))
 page.on('response',r=>{if(r.url().endsWith('.glb'))loaded.push({url:r.url(),status:r.status()})})
 await page.goto('http://127.0.0.1:5180/lunar-city-review.html')
 await page.waitForFunction(()=>document.documentElement.dataset.worldReady==='true',null,{timeout:90000})
 const options=await page.locator('#leaders option').allTextContents()
 for(const key of ['lunar-city:review-worker:ci-repair-triage','lunar-city:model:arts-studio']){
  await page.selectOption('#leaders',key)
  if(key.includes('review-worker'))await page.selectOption('#animation','work')
  await page.evaluate(()=>new Promise(resolve=>{let frames=0;const settle=()=>++frames>=45?resolve():requestAnimationFrame(settle);settle()}))
  await page.screenshot({path:`docs/lunar-city/game/evidence/built-${key.split(':').at(-1)}.png`})
 }
 const manifest=await (await page.request.get('http://127.0.0.1:5180/lunar-city/v2-review/world-manifest.v2.json')).json()
 const cat=manifest.models.find(m=>m.id==='arts-studio')
 const ci=manifest.reviewWorkerAssets.find(m=>m.id==='ci-repair-triage')
 const bytes=await (await page.request.get('http://127.0.0.1:5180/lunar-city/v2-review/'+cat.uri)).body()
 const sha=createHash('sha256').update(bytes).digest('hex')
 if(sha!==cat.statistics.sha256||!loaded.some(r=>r.url.endsWith(cat.uri)&&r.status===200)||!loaded.some(r=>r.url.endsWith(ci.uri)&&r.status===200))throw Error('Built assets did not load or SHA mismatch')
 if(errors.length)throw Error(errors.join('\n'))
 const receipt={options,errors,ready:true,loadedGlbs:loaded,cat:{uri:cat.uri,sha256:sha,bounds:cat.bounds,transform:cat.transform,heightMetres:cat.bounds.max[1]-cat.bounds.min[1]},ci:{uri:ci.uri},capturedAt:new Date().toISOString(),scope:'Actual production renderer build in headless WebGL; not packaged Electron or authenticated gateway acceptance'}
 await writeFile('docs/lunar-city/game/evidence/built-viewer.json',JSON.stringify(receipt,null,2));console.log({options,cat:receipt.cat,loaded:loaded.length})
}finally{await browser.close()}
