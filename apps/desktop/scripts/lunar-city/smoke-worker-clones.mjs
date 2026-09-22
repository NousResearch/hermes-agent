import { chromium } from '@playwright/test'
import { writeFile } from 'node:fs/promises'
const workerId = process.argv[2] ?? 'baseline'
const evidencePath = process.argv[3] ?? 'docs/lunar-city/game/evidence/worker-clones.json'
const browser = await chromium.launch({ headless: true })
try {
  const page = await browser.newPage()
  const errors = []
  page.on('pageerror', e => errors.push(e.message))
  await page.goto('http://127.0.0.1:5178/lunar-city-review.html?lunarCityReviewWorkers=baseline')
  await page.waitForFunction(() => document.documentElement.dataset.worldReady === 'true', null, { timeout: 90000 })
  await page.click('#motion')
  const result = await page.evaluate(async workerId => {
    const { loadBabylonModules } = await import('/src/app/lunar-city/world/create-world.ts')
    const { cloneWorkerHierarchy } = await import('/src/app/lunar-city/world/worker-hierarchy.ts')
    const modules = await loadBabylonModules(),
      scene = modules.Engine.Instances[0].scenes[0]
    const source = scene.meshes.find(m => m.skeleton && m.metadata?.lunarCity?.workerReviewId === workerId)
    let root = source
    while (root && root.name !== '__root__') root = root.parent
    if (!root) throw Error('Missing glTF root')
    const groups = new Map(
      scene.animationGroups
        .filter(g => g.targetedAnimations.some(t => t.target.metadata?.lunarCity?.workerReviewId === workerId))
        .map(g => [g.name, g])
    )
    const anchor = new modules.TransformNode('clone-audit', scene)
    const a = cloneWorkerHierarchy(root, anchor, groups, 'a'),
      b = cloneWorkerHierarchy(root, anchor, groups, 'b')
    const ma = a.nodeMap.get(source),
      mb = b.nodeMap.get(source)
    if (ma.skeleton === mb.skeleton || ma.skeleton === source.skeleton) throw Error('Shared skeleton')
    const base = Array.from(mb.getPositionData(true, true))
    const walk = a.animations.get('walk')
    walk.start(false)
    walk.pause()
    walk.goToFrame(15)
    scene.render()
    const moved = ma.getPositionData(true, true),
      stationary = mb.getPositionData(true, true)
    let displacement = 0,
      crosstalk = 0
    for (let i = 0; i < base.length; i++) {
      displacement = Math.max(displacement, Math.abs(moved[i] - base[i]))
      crosstalk = Math.max(crosstalk, Math.abs(stationary[i] - base[i]))
    }
    if (displacement < 0.001 || crosstalk > 1e-6) throw Error(`Motion ${displacement}, cross-talk ${crosstalk}`)
    const receipt = {
      bones: ma.skeleton.bones.length,
      independentSkeletons: true,
      maximumCoordinateDisplacementMetres: displacement,
      crossTalkMetres: crosstalk
    }
    for (const clone of [a, b]) {
      for (const group of clone.animations.values()) group.dispose()
      clone.disposeSkeletons()
      clone.root.dispose()
    }
    anchor.dispose()
    const {loadWorldManifest}=await import('/src/app/lunar-city/manifest.ts')
    const {createBabylonEntityFactory}=await import('/src/app/lunar-city/world/world-scene.ts')
    const manifest=await loadWorldManifest('/lunar-city/v2-review/world-manifest.v2.json')
    const model=manifest.models.find(m=>m.id==='workers')
    const legacy=await modules.ImportMeshAsync('/lunar-city/v2-review/'+model.uri,scene)
    const reviewed={meshes:[root,...root.getChildMeshes()],transformNodes:root.getDescendants().filter(n=>!n.getTotalVertices),animationGroups:[...groups.values()]}
    const factory=createBabylonEntityFactory(model,legacy,modules,scene,manifest.characterAssets,reviewed)
    const entity={key:'session:clone-audit',identity:{connectionId:'local',kind:'session',profile:'worker',sessionId:'clone-audit'},animation:'work',authority:'authoritative',destination:'project',observedAt:1}
    const visual=factory.createAnimated(entity,'builder')
    visual.setPosition({x:0,y:0,z:0});visual.setLod(0);visual.setAnimation('walk')
    const owned=scene.meshes.filter(m=>m.metadata?.lunarCity?.entityKey===entity.key)
    if(!owned.some(m=>m.skeleton&&m.isEnabled()))throw Error('Opt-in near worker missing')
    visual.setLod(1);visual.setAnimation('walk')
    if(owned.some(m=>m.skeleton&&m.isEnabled()))throw Error('Review skin visible at mid LOD')
    if(!owned.some(m=>!m.skeleton&&m.getTotalVertices()>0&&m.isEnabled()))throw Error('Historical mid LOD fallback missing')
    visual.setLod(0);visual.setAnimation('work')
    if(!owned.some(m=>m.skeleton&&m.isEnabled()))throw Error('Near skin did not return')
    visual.dispose()
    receipt.nearMidNearTransitions=true
    return receipt
  }, workerId)
  if (errors.length) throw Error(errors.join('\n'))
  await writeFile(
    evidencePath,
    JSON.stringify({ ...result, pageErrors: errors, capturedAt: new Date().toISOString() }, null, 2)
  )
  console.log(result)
} finally {
  await browser.close()
}
