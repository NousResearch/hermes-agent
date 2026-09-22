import { chromium } from '@playwright/test'
import { mkdir, writeFile } from 'node:fs/promises'
import { createHash } from 'node:crypto'
import assert from 'node:assert/strict'
import { resolve } from 'node:path'
import { REVIEW_BUILDINGS } from './build-review-buildings.mjs'
import { BUILDING_SCALE } from './settlement-layout.mjs'

const url = process.argv[2] ?? 'http://127.0.0.1:5180/lunar-city-review.html'
const output = resolve(process.argv[3] ?? 'docs/lunar-city/game/evidence/building-roster')
await mkdir(output, { recursive: true })
const browser = await chromium.launch({ headless: true })
try {
  const page = await browser.newPage({ viewport: { width: 1600, height: 1100 } })
  const errors = [], loaded = []
  page.on('pageerror', error => errors.push(error.message))
  page.on('response', response => {
    if (response.url().endsWith('.glb')) loaded.push({ url: response.url(), status: response.status() })
  })
  await page.goto(url)
  await page.waitForFunction(() => document.documentElement.dataset.worldReady === 'true' || !document.querySelector('#retry').hidden, null, { timeout: 90000 })
  if (await page.locator('#retry').isVisible()) throw Error(await page.locator('body').innerText())
  const pack = new URL('/lunar-city/v2-review/', url)
  const manifest = await (await page.request.get(new URL('world-manifest.v2.json', pack).href)).json()
  assert.equal(new Set(manifest.models.map(model => model.id)).size, manifest.models.length)
  const options = await page.locator('#leaders option').evaluateAll(nodes => nodes.map(node => node.value))
  const buildings = []
  for (const [species, id, , height] of REVIEW_BUILDINGS) {
    const models = manifest.models.filter(model => model.id === id)
    assert.equal(models.length, 1)
    const model = models[0], key = `lunar-city:model:${id}`
    assert.equal(options.filter(option => option === key).length, 1)
    assert.ok(model.uri.startsWith(`models/review-building-${species}-`))
    assert.ok(loaded.some(response => response.url === new URL(model.uri, pack).href && response.status === 200))
    const bytes = await (await page.request.get(new URL(model.uri, pack).href)).body()
    const sha256 = createHash('sha256').update(bytes).digest('hex')
    assert.equal(sha256, model.statistics.sha256)
    const gltf = JSON.parse(bytes.subarray(20, 20 + bytes.readUInt32LE(12)))
    const positions = gltf.meshes.flatMap(mesh => mesh.primitives.map(primitive => gltf.accessors[primitive.attributes.POSITION]))
    const loadedHeight = Math.max(...positions.map(accessor => accessor.max[1])) - Math.min(...positions.map(accessor => accessor.min[1]))
    assert.ok(Math.abs(loadedHeight - height) < .005)
    assert.deepEqual(model.transform.scale, [BUILDING_SCALE, BUILDING_SCALE, BUILDING_SCALE])
    await page.selectOption('#leaders', key)
    await page.evaluate(() => new Promise(resolve => {
      let count = 0
      const frame = () => ++count >= 60 ? resolve() : requestAnimationFrame(frame)
      requestAnimationFrame(frame)
    }))
    await page.screenshot({ path: resolve(output, `${species}-${id}.png`) })
    buildings.push({ species, id, uri: model.uri, sha256, sourceHeightMetres: loadedHeight, placedHeightMetres: loadedHeight * model.transform.scale[1], position: model.transform.position })
  }
  await page.click('#overview')
  await page.evaluate(() => new Promise(resolve => {
    let count = 0
    const frame = () => ++count >= 60 ? resolve() : requestAnimationFrame(frame)
    requestAnimationFrame(frame)
  }))
  await page.screenshot({ path: resolve(output, 'overview.png') })
  assert.deepEqual(errors, [])
  const receipt = { url, capturedAt: new Date().toISOString(), scope: 'Actual built viewer request/load, source GLB bounds, declared town design transforms, unique focus options and screenshots; visual quality remains review only', buildings, errors }
  await writeFile(resolve(output, 'receipt.json'), JSON.stringify(receipt, null, 2) + '\n')
  console.log({ pass: true, buildings: buildings.length, output })
} finally {
  await browser.close()
}
