import { chromium } from '@playwright/test'
import { writeFile } from 'node:fs/promises'
import { resolve } from 'node:path'
import { PEDESTRIAN_ROUTES, obstacles, segmentBlocked } from './settlement-layout.mjs'
const browser = await chromium.launch({ headless: true })
try {
  const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } })
  await page.goto('http://127.0.0.1:5178/lunar-city-review.html')
  await page.waitForFunction(
    () => document.documentElement.dataset.worldReady === 'true' || !document.querySelector('#retry').hidden,
    null,
    { timeout: 60000 }
  )
  if (await page.locator('#retry').isVisible()) throw new Error(await page.locator('body').innerText())
  await page.click('#motion')
  const result = await page.evaluate(async routes => {
    const { loadBabylonModules } = await import('/src/app/lunar-city/world/create-world.ts')
    const { loadWorldManifest } = await import('/src/app/lunar-city/manifest.ts')
    const { createRouteNavigationQuery } = await import('/src/app/lunar-city/world/world-navigation.ts')
    const modules = await loadBabylonModules(),
      scene = modules.Engine.Instances[0].scenes[0]
    const manifest = await loadWorldManifest('/lunar-city/v2-review/world-manifest.v2.json')
    const query = await createRouteNavigationQuery(
      manifest.navigation,
      modules,
      scene,
      uri => new URL(`/lunar-city/v2-review/${uri}`, location.href).href
    )
    const vec = p => ({ x: p[0], y: p[1], z: p[2] })
    const paths = []
    for (const route of routes) {
      const from = vec(route.points[0]),
        to = vec(route.points.at(-1))
      // A non-node origin proves the actual Recast path, not the exact-node fallback graph.
      from.x += 0.02
      const path = query.computePath(from, to)
      if (!path?.length) throw new Error(`No real navmesh route: ${route.from} -> ${route.to}`)
      const end = path.at(-1)
      const arrivalError = Math.hypot(end.x - to.x, end.z - to.z)
      if (arrivalError > 0.6) throw new Error(`Partial navmesh route ${route.from} -> ${route.to}: ${arrivalError}m`)
      paths.push({ from: route.from, to: route.to, path, arrivalError })
    }
    query.dispose?.()
    return { coordinates: scene.useRightHandedSystem ? 'right-handed' : 'left-handed', paths }
  }, PEDESTRIAN_ROUTES)
  let maxGroundError = 0
  for (const route of result.paths)
    for (let index = 0; index < route.path.length; index++) {
      const p = route.path[index]
      let nearest = Infinity,
        error = Infinity
      for (const authored of PEDESTRIAN_ROUTES)
        for (let i = 1; i < authored.points.length; i++) {
          const a = authored.points[i - 1],
            b = authored.points[i],
            dx = b[0] - a[0],
            dz = b[2] - a[2]
          const t = Math.max(0, Math.min(1, ((p.x - a[0]) * dx + (p.z - a[2]) * dz) / (dx * dx + dz * dz)))
          const distance = Math.hypot(p.x - a[0] - dx * t, p.z - a[2] - dz * t)
          if (distance < nearest) {
            nearest = distance
            error = Math.abs(p.y - a[1] - (b[1] - a[1]) * t)
          }
        }
      maxGroundError = Math.max(maxGroundError, error)
      if (error > 0.02) throw new Error(`Navigation floats ${error}m above the authored road`)
      if (index)
        for (const obstacle of obstacles) {
          const a = route.path[index - 1]
          if (
            segmentBlocked([a.x, a.y, a.z], [p.x, p.y, p.z], {
              ...obstacle,
              halfWidth: obstacle.halfWidth - 1.3,
              halfDepth: obstacle.halfDepth - 1.3
            })
          )
            throw new Error(`Navmesh crosses ${obstacle.id}`)
        }
    }
  result.maximumGroundErrorMetres = maxGroundError
  await writeFile(
    resolve('docs/lunar-city/game/evidence/navigation.json'),
    JSON.stringify({ capturedAt: new Date().toISOString(), ...result }, null, 2) + '\n'
  )
  console.log(
    JSON.stringify({
      coordinates: result.coordinates,
      routes: result.paths.length,
      maxGroundError,
      maxArrivalError: Math.max(...result.paths.map(p => p.arrivalError))
    })
  )
} finally {
  await browser.close()
}
