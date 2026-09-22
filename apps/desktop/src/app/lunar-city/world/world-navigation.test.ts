// @vitest-environment node
import { createHash } from 'node:crypto'

import { expect, it, vi } from 'vitest'

import type { LunarCityWorldModules, WorldManifestV2 } from '../model'
import { importRuntimeAsset, RuntimeAssetIntegrityError } from '../runtime-asset-integrity'

import { createRouteNavigationQuery } from './world-navigation'

it('propagates actual navigation byte corruption before decoding instead of returning a traversable fallback', async () => {
  const from = { x: 0, y: 0, z: 0 }, to = { x: 1, y: 0, z: 0 }
  const navigation = { meshUri: 'navigation.glb', links: [{ from, to, bidirectional: true }], areas: [] } as WorldManifestV2['navigation']
  const url = 'https://city.test/navigation.glb'
  const valid = new TextEncoder().encode('expected navigation bytes')
  const digests = new Map([[url, createHash('sha256').update(valid).digest('hex')]])
  const decode = vi.fn()

  const modules = {
    ImportMeshAsync: () => importRuntimeAsset(url, digests, decode, { fetch: async () => new Response('corrupted') }),
    createRecastNavigation: vi.fn()
  } as unknown as LunarCityWorldModules

  await expect(createRouteNavigationQuery(navigation, modules, {} as never, () => url)).rejects.toBeInstanceOf(RuntimeAssetIntegrityError)
  expect(decode).not.toHaveBeenCalled()
  expect(modules.createRecastNavigation).not.toHaveBeenCalled()
})

it('holds movement when navigation geometry or Recast initialization cannot establish a floor', async () => {
  const from = { x: 0, y: 0, z: 0 }, to = { x: 1, y: 0, z: 0 }
  const navigation = { meshUri: 'navigation.glb', links: [{ from, to, bidirectional: true }], areas: [] } as WorldManifestV2['navigation']
  const dispose = vi.fn()

  const modules = {
    ImportMeshAsync: async () => ({ meshes: [{ getVerticesData: () => [0,0,0,1,0,0,0,0,1], getIndices: () => [0,1,2], dispose }], transformNodes: [] }),
    createRecastNavigation: async () => { throw new Error('WASM unavailable') }
  } as unknown as LunarCityWorldModules

  const query = await createRouteNavigationQuery(navigation, modules, {} as never, uri => uri)
  expect(query.computePath(from, to)).toBeUndefined()
  expect(query.resolvePosition?.(from)).toBeUndefined()
  expect(query.canTraverse?.(from, to)).toBe(false)
  expect(query.computePath({ ...from, x: .02 }, to)).toBeUndefined()
  expect(dispose).toHaveBeenCalledOnce()
})
