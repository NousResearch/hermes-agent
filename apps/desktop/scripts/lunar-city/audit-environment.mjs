import { chromium } from '@playwright/test'
import { writeFile } from 'node:fs/promises'
const browser = await chromium.launch({ headless: true })
try {
  const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } })
  page.on('pageerror', e => console.error(e.message))
  await page.goto('http://127.0.0.1:5178/lunar-city-review.html')
  try {
    await page.waitForFunction(
      () => document.documentElement.dataset.worldReady === 'true' || !document.querySelector('#retry').hidden,
      null,
      { timeout: 60000 }
    )
  } catch (error) {
    console.error(await page.locator('body').innerText())
    throw error
  }
  if (await page.locator('#retry').isVisible()) throw new Error(await page.locator('body').innerText())
  await page.click('#motion')
  const audit = await page.evaluate(async () => {
    const { loadBabylonModules } = await import('/src/app/lunar-city/world/create-world.ts')
    const modules = await loadBabylonModules()
    const scene = modules.Engine.Instances[0].scenes[0]
    return {
      meshes: scene.meshes
        .filter(m => m.getTotalVertices() > 0)
        .map(m => ({
          name: m.name,
          metadata: m.metadata,
          min: m.getBoundingInfo().boundingBox.minimumWorld.asArray(),
          max: m.getBoundingInfo().boundingBox.maximumWorld.asArray(),
          material: m.material?.name
        })),
      materials: scene.materials.map(m => ({
        name: m.name,
        color: m.albedoColor?.asArray(),
        metallic: m.metallic,
        roughness: m.roughness
      })),
      lights: scene.lights.map(l => ({ name: l.name, intensity: l.intensity, diffuse: l.diffuse.asArray() }))
    }
  })
  await writeFile('/private/tmp/lunar-environment-audit.json', JSON.stringify(audit, null, 2))
  await page.screenshot({ path: '/private/tmp/lunar-environment-overview.png' })
  console.log('Recorded loaded geometry, lighting, and overview')
} finally {
  await browser.close()
}
