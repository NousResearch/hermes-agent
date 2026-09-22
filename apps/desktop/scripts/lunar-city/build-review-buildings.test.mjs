import { test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtemp, mkdir, readFile, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { buildReviewBuildings, REVIEW_BUILDINGS } from './build-review-buildings.mjs'
import { BUILDING_SCALE } from './settlement-layout.mjs'

const assets = resolve(dirname(fileURLToPath(import.meta.url)), '../../public/lunar-city')
test('card buildings replace their role once and retain source metre dimensions on repeated assembly', async () => {
  const output = await mkdtemp(resolve(tmpdir(), 'lunar-building-contract-'))
  try {
    await mkdir(resolve(output, 'models'))
    const manifest = JSON.parse(await readFile(resolve(assets, 'v2/world-manifest.v2.json'), 'utf8'))
    await buildReviewBuildings(manifest, assets, output, [])
    const count = manifest.models.length
    await buildReviewBuildings(manifest, assets, output, [])
    assert.equal(manifest.models.length, count)
    for (const [species, id, , height] of REVIEW_BUILDINGS) {
      const models = manifest.models.filter(model => model.id === id)
      assert.equal(models.length, 1)
      const model = models[0]
      assert.ok(model.uri.startsWith(`models/review-building-${species}-`))
      assert.ok(Math.abs(model.bounds.max[1] - model.bounds.min[1] - height) < .005)
      assert.deepEqual(model.transform.scale, [BUILDING_SCALE, BUILDING_SCALE, BUILDING_SCALE])
      assert.ok((await readFile(resolve(output, model.uri))).byteLength > 0)
    }
  } finally {
    await rm(output, { recursive: true })
  }
})
test('duplicate role input is rejected before any asset write', async () => {
  const manifest = { models: [{ id: 'library' }, { id: 'library' }] }
  await assert.rejects(() => buildReviewBuildings(manifest, assets, '/does-not-exist', []), /duplicate model roles/)
})
