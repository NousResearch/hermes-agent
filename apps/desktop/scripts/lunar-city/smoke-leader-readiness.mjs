import { chromium } from '@playwright/test'
import { mkdir, writeFile, readFile } from 'node:fs/promises'
import { resolve } from 'node:path'
const url=process.argv[2]??'http://127.0.0.1:5178/lunar-city-review.html'
const output=resolve(process.argv[3]??'docs/lunar-city/game/evidence/leader-readiness.json')
const browser=await chromium.launch({headless:true})
try {
  const page=await browser.newPage()
  await page.route('**/walk-contact-review/rigged-review.glb',async route=>{
    const path=resolve('apps/desktop/public',new URL(route.request().url()).pathname.slice(1))
    await route.fulfill({body:await readFile(path),contentType:'model/gltf-binary'})
  })
  await page.goto(url)
  await page.waitForFunction(()=>document.documentElement.dataset.worldReady==='true',null,{timeout:120000})
  const receipt=await page.evaluate(async(catOnly)=>{
    const {loadBabylonModules}=await import('/src/app/lunar-city/world/create-world.ts')
    const {importReviewLeader}=await import('/src/app/lunar-city/world/review-assets.ts')
    const modules=await loadBabylonModules()
    const {loadWorldManifest}=await import('/src/app/lunar-city/manifest.ts')
    const manifest=await loadWorldManifest('/lunar-city/v2-review/world-manifest.v2.json')
    const engine=modules.Engine.Instances.at(-1)
    const directories={owl:'owl-librarian',elephant:'elephant-memory',cat:'cat-arts',fox:'fox-scientist',capybara:'capybara-revenue',lion:'lion-steward',beaver:'beaver-architect',monkey:'monkey-poet'}
    const results=[]
    const baselineFrames=[]
    let restMapping
    for(const asset of manifest.reviewLeaderAssets.filter(asset=>!catOnly||asset.id==='cat')) {
      for(const variant of ['manifest',...(catOnly?[]:['held-rig-source']),...(asset.id==='monkey'?[]:['contact-review'])]) {
        const scene=new modules.Scene(engine)
        try {
          const uri=variant==='manifest'?new URL(asset.uri,location.origin+'/lunar-city/v2-review/').href:location.origin+'/lunar-city/multiview-2026-09-07/'+directories[asset.id]+'/'+(variant==='contact-review'?'walk-contact-review/':'')+'rigged-review.glb'
          const imported=await importReviewLeader({...asset,uri},scene,modules,value=>value)
          let contact
          if(variant==='contact-review'||(asset.id==='cat'&&variant==='manifest')) {
            const walk=imported.groups.get('walk')
            const samples=[]
            let normalizedMaximumDifference=0
            if(walk?.hasMotion===true) {
              walk.start(true);walk.pause()
              for(let index=0;index<=40;index++) {
                const frame=walk.from+(walk.to-walk.from)*index/40
                walk.goToFrame(frame)
                for(const node of imported.result.transformNodes)node.computeWorldMatrix?.(true)
                for(const skeleton of imported.result.skeletons??[])skeleton.prepare(true)
                let minimumY=Infinity,maximumY=-Infinity
                const deformed=[]
                for(const mesh of imported.result.meshes) {
                  if(!mesh.getTotalVertices())continue
                  mesh.computeWorldMatrix(true);mesh.skeleton?.prepare(true)
                  const vertices=mesh.getPositionData(true,true),m=mesh.getWorldMatrix().m
                  for(let offset=0;offset<vertices.length;offset+=3) {
                    const y=m[1]*vertices[offset]+m[5]*vertices[offset+1]+m[9]*vertices[offset+2]+m[13]-asset.position.y
                    minimumY=Math.min(minimumY,y);maximumY=Math.max(maximumY,y)
                    if(catOnly)deformed.push(m[0]*vertices[offset]+m[4]*vertices[offset+1]+m[8]*vertices[offset+2]+m[12]-asset.position.x,y,m[2]*vertices[offset]+m[6]*vertices[offset+1]+m[10]*vertices[offset+2]+m[14]-asset.position.z)
                  }
                }
                if(catOnly&&variant==='manifest')baselineFrames[index]=new Float32Array(deformed)
                if(catOnly&&variant==='contact-review') {
                  const baseline=baselineFrames[index]
                  if(index===0) {
                    const baselineMin=[Infinity,Infinity,Infinity],copyMin=[Infinity,Infinity,Infinity]
                    for(let n=0;n<baseline.length;n++)baselineMin[n%3]=Math.min(baselineMin[n%3],baseline[n])
                    for(let n=0;n<deformed.length;n++)copyMin[n%3]=Math.min(copyMin[n%3],deformed[n])
                    const shift=copyMin.map((value,axis)=>value-baselineMin[axis]),grid=new Map(),cell=.00001
                    for(let n=0;n<baseline.length;n+=3) {
                      const key=[0,1,2].map(axis=>Math.floor(baseline[n+axis]/cell)).join(':')
                      const values=grid.get(key)??[];values.push(n);grid.set(key,values)
                    }
                    restMapping=[]
                    for(let n=0;n<deformed.length;n+=3) {
                      const point=[0,1,2].map(axis=>deformed[n+axis]-shift[axis]),bin=point.map(value=>Math.floor(value/cell))
                      let nearest=-1,best=cell*2
                      for(let x=-1;x<=1;x++)for(let y=-1;y<=1;y++)for(let z=-1;z<=1;z++)for(const candidate of grid.get([bin[0]+x,bin[1]+y,bin[2]+z].join(':'))??[]) {
                        const d=Math.hypot(...[0,1,2].map(axis=>point[axis]-baseline[candidate+axis]))
                        if(d<best){nearest=candidate;best=d}
                      }
                      if(nearest<0)throw new Error('Contact copy rest vertex did not match within20 micrometres at '+n)
                      restMapping.push(nearest)
                    }
                  }
                  const shift=[0,0,0],count=deformed.length/3
                  for(let vertex=0;vertex<deformed.length;vertex++)shift[vertex%3]+=(deformed[vertex]-baseline[restMapping[Math.floor(vertex/3)]+vertex%3])/count
                  for(let vertex=0;vertex<deformed.length;vertex+=3)normalizedMaximumDifference=Math.max(normalizedMaximumDifference,Math.hypot(...[0,1,2].map(axis=>deformed[vertex+axis]-baseline[restMapping[vertex/3]+axis]-shift[axis])))
                }
                samples.push({frame,minimumY,maximumY})
              }
              walk.stop()
            }
            contact={samples,normalizedMaximumDifference:catOnly&&variant==='contact-review'?normalizedMaximumDifference:undefined,minimumY:Math.min(...samples.map(sample=>sample.minimumY)),maximumGroundGap:Math.max(...samples.map(sample=>sample.minimumY)),scope:'Dense whole-mesh CPU-skinned world ground minimum at41 walk frames; does not identify individual feet or validate deformation aesthetics.'}
          }
          results.push({id:asset.id,variant,uri,contact,canWalk:imported.groups.get('walk')?.hasMotion===true,groups:[...imported.groups].map(([state,group])=>({state,name:group.name,hasMotion:group.hasMotion})),skinnedMeshes:imported.result.meshes.filter(mesh=>mesh.skeleton).length})
        } finally {scene.dispose()}
      }
    }
    return {results,scope:'Actual Babylon import and weighted-target animation motion detection. Clip availability only; no visual deformation acceptance or source-rig promotion.'}
  },process.argv.includes('--cat-only'))
  await mkdir(resolve(output,'..'),{recursive:true})
  await writeFile(output,JSON.stringify({capturedAt:new Date().toISOString(),url,...receipt},null,2))
  console.log(JSON.stringify(receipt.results.map(({id,variant,canWalk})=>({id,variant,canWalk}))))
} finally {await browser.close()}
