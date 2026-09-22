/* global document, Option */
import { chromium } from '@playwright/test'
import { mkdir, writeFile } from 'node:fs/promises'
import { resolve } from 'node:path'

const origin = process.argv[2] ?? 'http://127.0.0.1:5178'
const output = resolve(process.argv[3] ?? 'docs/lunar-city/game/evidence/loaded-historical-bounds.json')
const browser = await chromium.launch({ headless: true })
const receipts = []
try {
  for (const pack of ['review', 'historical']) {
    const page = await browser.newPage({ viewport: { width: 1600, height: 1100 } })
    console.log(`Loading ${pack}`)
    const errors = []
    page.on('pageerror', e => errors.push(e.message))
    if (pack === 'historical') await page.route('**/lunar-city/v2-review/**', async route => {
      const response = await route.fetch({ url: route.request().url().replace('/lunar-city/v2-review/', '/lunar-city/v2/') })
      await route.fulfill({ response })
    })
    await page.goto(`${origin}/lunar-city-review.html`)
    await page.waitForFunction(() => document.documentElement.dataset.worldReady === 'true' || document.querySelector('#retry')?.hidden === false, null, { timeout: 90000 })
    if (await page.getAttribute('html', 'data-world-ready') !== 'true') throw new Error(await page.locator('#loading').innerText())
    console.log(`Measuring ${pack}`)
    // Inspect the active representation after actual scheduler placement; no mesh/transform mutation.
    const measure = () => page.evaluate(async () => {
      const { Engine } = await (await import('/src/app/lunar-city/world/create-world.ts')).loadBabylonModules()
      const scene = Engine.Instances.at(-1).scenes.at(-1)
      const manifests = await Promise.all(['v2', 'v2-review'].map(async name => [name, await (await fetch(`/lunar-city/${name}/world-manifest.v2.json`)).json()]))
      const models = manifests[0][1].models
      const groups = new Map()
      let untaggedVisible = 0
      for (const mesh of scene.meshes) {
        const metadata = mesh.metadata?.lunarCity
        if (!metadata?.modelId) { if (mesh.isEnabled() && mesh.isVisible && mesh.visibility > 0 && mesh.getTotalVertices() > 0) untaggedVisible++; continue }
        const owner = metadata.entityKey ?? metadata.leaderId ?? ''
        const key = JSON.stringify([metadata.modelId, owner])
        const group = groups.get(key) ?? { modelId: metadata.modelId, owner, kind: metadata.kind, visibleMeshes: 0, hiddenMeshes: 0, min: { x: Infinity, y: Infinity, z: Infinity }, max: { x: -Infinity, y: -Infinity, z: -Infinity }, roots: new Set() }
        groups.set(key, group)
        if (!mesh.isEnabled() || !mesh.isVisible || mesh.visibility <= 0 || mesh.getTotalVertices() <= 0) { group.hiddenMeshes++; continue }
        mesh.computeWorldMatrix(true)
        const bounds = mesh.getBoundingInfo().boundingBox
        for (const axis of ['x', 'y', 'z']) { group.min[axis] = Math.min(group.min[axis], bounds.minimumWorld[axis]); group.max[axis] = Math.max(group.max[axis], bounds.maximumWorld[axis]) }
        let root = mesh
        while (root.parent) root = root.parent
        group.roots.add(root.name)
        group.visibleMeshes++
      }
      return { groups: [...groups.values()].map(group => ({ ...group, roots: [...group.roots], min: group.visibleMeshes ? group.min : null, max: group.visibleMeshes ? group.max : null, dimensions: group.visibleMeshes ? Object.fromEntries(['x','y','z'].map(axis => [axis, group.max[axis]-group.min[axis]])) : null })), models: models.map(model => ({ id: model.id, uri: model.uri, bounds: model.bounds, scale: model.transform.scale, targetDimensions: Object.fromEntries(['x','y','z'].map((axis, index) => [axis, (model.bounds.max[index]-model.bounds.min[index])*Math.abs(model.transform.scale[index])])) })), untaggedVisible, sceneMeshes: scene.meshes.length }
    })
    const overview = await measure()
    const focused = []
    const options = await page.evaluate(async pack => {
      const manifest = await (await fetch(`/lunar-city/${pack === 'historical' ? 'v2' : 'v2-review'}/world-manifest.v2.json`)).json()
      const keys = manifest.models.map(model => `lunar-city:model:${model.id}`)
      for (const leader of manifest.characterAssets.leaders) keys.push(`lunar-city:leader:${leader.id}`)
      const select = document.querySelector('#leaders')
      for (const key of keys) if (![...select.options].some(option => option.value === key)) select.add(new Option(key,key))
      return keys
    },pack)
    for (const key of options) {
      await page.selectOption('#leaders', key)
      await page.waitForTimeout(150)
      const capture = await measure()
      const id = key.split(':').at(-1)
      focused.push({ id, focusKey:key, groups: capture.groups.filter(group => key.includes(':leader:') ? group.modelId === 'leaders' : group.modelId === id) })
    }
    await page.click('#overview')
    await page.getByRole('button', { name: 'Local systems demo', exact:true }).click()
    await page.waitForTimeout(300)
    const demo = await measure()
    const focusedWorkers = []
    for (const group of demo.groups.filter(group => group.modelId === 'workers' && group.owner)) {
      await page.evaluate(key => { const select = document.querySelector('#leaders'); select.add(new Option(key,key)) },group.owner)
      await page.selectOption('#leaders',group.owner)
      await page.waitForTimeout(200)
      focusedWorkers.push(...(await measure()).groups.filter(candidate => candidate.owner === group.owner))
    }
    const sampleAt = new Date().toISOString()
    receipts.push({ sampleAt, focusedWorkers, demoWorkers:demo.groups.filter(group => group.modelId==='workers'), pack, manifestOverride: pack === 'historical' ? 'Original v2 manifest and corresponding bytes loaded through browser-only v2-review request redirection; unchanged manifest contents' : null, overview, focused, errors })
    await mkdir(resolve(output, '..'), { recursive: true })
    await writeFile(output, JSON.stringify({ capturedAt: new Date().toISOString(), origin, receipts }, null, 2))
    await page.close()
  }
  await mkdir(resolve(output, '..'), { recursive: true })
  await writeFile(output, JSON.stringify({ capturedAt: new Date().toISOString(), origin, evidence: 'Actual loaded WebGL scene bounds; enabled/isVisible/visibility-positive geometry only. Hidden template/LOD meshes counted separately. Not packaged acceptance.', receipts }, null, 2))
  console.log(output)
} finally { await browser.close() }
