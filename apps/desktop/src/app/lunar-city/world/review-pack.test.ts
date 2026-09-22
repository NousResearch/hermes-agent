import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

import { describe, expect, it } from 'vitest'

import { parseWorldManifest } from '../manifest'

const root = resolve('public/lunar-city/v2-review')
describe('assembled review pack', () => {
  it('loads through the production manifest parser with every declared asset present', () => {
    const manifest = parseWorldManifest(JSON.parse(readFileSync(resolve(root, 'world-manifest.v2.json'), 'utf8')))

    for (const uri of [
      ...manifest.models.map(model => model.uri),
      manifest.navigation.meshUri,
      ...manifest.reviewLeaderAssets!.map(asset => asset.uri)
    ]) {
      expect(readFileSync(resolve(root, uri)).byteLength).toBeGreaterThan(0)
    }

    const points = new Set(
      manifest.navigation.links.flatMap(link => [JSON.stringify(link.from), JSON.stringify(link.to)])
    )

    for (const point of Object.values(manifest.destinations)) {
      expect(points.has(JSON.stringify(point))).toBe(true)
    }
  })
})
