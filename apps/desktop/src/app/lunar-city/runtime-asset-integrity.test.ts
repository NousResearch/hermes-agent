// @vitest-environment node
import { createHash } from 'node:crypto'

import { describe, expect, it, vi } from 'vitest'

import type { ModelManifestEntry, WorldManifestV2 } from './model'
import { buildRuntimeAssetDigests, importRuntimeAsset } from './runtime-asset-integrity'

const bytes = new TextEncoder().encode('verified asset bytes')
const sha256 = createHash('sha256').update(bytes).digest('hex')
const manifestUrl = 'https://city.test/v2/world-manifest.v2.json'
const assetUrl = 'https://city.test/v2/models/worker.glb'

function manifest(): Pick<WorldManifestV2, 'models' | 'reviewLeaderAssets' | 'reviewWorkerAssets'> {
  return { models: [{ uri: 'models/worker.glb', statistics: { sha256 } } as ModelManifestEntry] }
}

describe('runtime asset integrity', () => {
  it('binds only declared paths to consistent receipts and refuses missing review hashes', () => {
    const review = {
      id: 'baseline',
      uri: 'models/review.glb',
      heightMetres: 1.2,
      position: { x: 0, y: 0, z: 0 },
      rotationY: 0
    }

    const input = { ...manifest(), reviewWorkerAssets: [review] }
    expect(() => buildRuntimeAssetDigests(input, manifestUrl)).toThrow(/receipts are required/)
    expect(() => buildRuntimeAssetDigests(input, manifestUrl, [])).toThrow(/receipt is missing/)

    const map = buildRuntimeAssetDigests(input, manifestUrl, [
      { uri: review.uri, sha256 },
      { uri: 'models/unlisted.glb', sha256: 'bad' }
    ])

    expect(map.get(assetUrl)).toBe(sha256)
    expect(map.has('https://city.test/v2/models/unlisted.glb')).toBe(false)
    expect(() =>
      buildRuntimeAssetDigests(input, manifestUrl, [
        { uri: review.uri, sha256 },
        { uri: review.uri, sha256: '0'.repeat(64) }
      ])
    ).toThrow(/conflicting digests/)
  })
  it('imports one fetch of verified bytes, rejects corruption before import, and revokes after importer failure', async () => {
    const map = buildRuntimeAssetDigests(manifest(), manifestUrl)
    const fetchAsset = vi.fn(async () => new Response(bytes))
    let importedUrl = ''

    const importer = vi.fn(async (url: string, extension?: '.glb') => {
      importedUrl = url
      expect(extension).toBe('.glb')
      expect(new Uint8Array(await (await fetch(url)).arrayBuffer())).toEqual(bytes)

      return 'loaded'
    })

    expect(await importRuntimeAsset(assetUrl, map, importer, { fetch: fetchAsset })).toBe('loaded')
    expect(fetchAsset).toHaveBeenCalledTimes(1)
    expect(importer).toHaveBeenCalledTimes(1)
    await expect(fetch(importedUrl)).rejects.toThrow()
    await expect(
      importRuntimeAsset(assetUrl, map, importer, {
        fetch: async () => new Response('corrupt')
      })
    ).rejects.toThrow(/digest mismatch/)
    expect(importer).toHaveBeenCalledTimes(1)
    await expect(
      importRuntimeAsset(
        assetUrl,
        map,
        async url => {
          importedUrl = url
          throw new Error('GLB decode failed')
        },
        { fetch: fetchAsset }
      )
    ).rejects.toThrow(/GLB decode failed/)
    await expect(fetch(importedUrl)).rejects.toThrow()
    const direct = vi.fn(async (url: string) => url)
    const navigation = 'https://city.test/v2/models/navigation.glb'
    expect(await importRuntimeAsset(navigation, map, direct)).toBe(navigation)
    expect(direct).toHaveBeenCalledWith(navigation)
  })
})
