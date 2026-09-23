import { chromium } from '@playwright/test'
import assert from 'node:assert/strict'
import { mkdir, writeFile } from 'node:fs/promises'
const browser = await chromium.launch({ headless: true })
try {
 const page = await browser.newPage({ viewport: { width: 1600, height: 1100 } })
 await page.goto('http://127.0.0.1:5178/lunar-city-review.html')
 await page.waitForFunction(() => document.documentElement.dataset.worldReady === 'true' || !document.querySelector('#retry').hidden, null, { timeout: 90000 })
  if (await page.locator('#retry').isVisible()) throw Error(await page.locator('body').innerText())
 const sample = () => page.evaluate(async () => {
  const { loadBabylonModules } = await import('/src/app/lunar-city/world/create-world.ts')
  const { Engine } = await loadBabylonModules(), scene = Engine.Instances[0].scenes[0]
  return { sun: scene.getMeshByName('lunar-city:backdrop-sun').position.asArray(), cloud: scene.getMeshByName('lunar-city:cloud:0').position.asArray(), key: scene.getLightByName('lunar-city:key-light').intensity, pickableBackdrop: scene.meshes.filter(mesh => /lunar-city:(ridge|sky|cloud|backdrop)/.test(mesh.name) && mesh.isPickable).map(mesh => mesh.name) }
 })
 const before = await sample(); await page.waitForTimeout(1200); const after = await sample()
 assert.notDeepEqual(after.sun, before.sun)
 assert.notDeepEqual(after.cloud, before.cloud)
 assert.deepEqual(after.pickableBackdrop, [])
 await page.click('#motion'); const paused = await sample(); await page.waitForTimeout(1200)
 assert.deepEqual(await sample(), paused)
 await page.click('#motion'); await page.waitForTimeout(1200); const resumed = await sample()
 assert.notDeepEqual(resumed.sun, paused.sun)
 await writeFile('docs/lunar-city/game/evidence/atmosphere-runtime.json', JSON.stringify({ before, after, paused, resumed, capturedAt: new Date().toISOString(), scope: 'Actual scene sun/cloud motion, frozen reduced-motion state, resume and nonpickable backdrop' }, null, 2))
 await page.click('#motion')
 await mkdir('docs/lunar-city/game/evidence/atmosphere', { recursive: true })
 for (const [name, phase] of [['day', .27], ['night', .77]]) {
  await page.evaluate(async phase => {
   const { loadBabylonModules } = await import('/src/app/lunar-city/world/create-world.ts')
   const { setAtmospherePresentationPhase } = await import('/src/app/lunar-city/world/atmosphere.ts')
   const { Engine } = await loadBabylonModules(), scene = Engine.Instances[0].scenes[0]
   setAtmospherePresentationPhase(scene, phase)
   scene.render()
  }, phase)
  await page.screenshot({ path: `docs/lunar-city/game/evidence/atmosphere/${name}.png` })
 }
 console.log('Atmosphere actual scene motion/pause/resume and controlled day/night screenshots passed')
} finally { await browser.close() }
