import { expect, it, vi } from 'vitest'

import { createBabylonPerfAdapter } from './perf-adapter'

it('reads draw calls and active triangles through one Babylon instrumentation adapter', () => {
  const dispose = vi.fn()

  const adapter = createBabylonPerfAdapter(
    {
      _activeIndices: { current: 3600 },
      meshes: [{}, {}],
      textures: [{}, {}, {}]
    },
    { dispose, drawCallsCounter: { current: 17 } }
  )

  expect(adapter.snapshot()).toEqual({ drawCalls: 17, entities: 2, textures: 3, visibleTriangles: 1200 })
  adapter.dispose()
  expect(dispose).toHaveBeenCalledOnce()
})

it('fails closed to zero when Babylon instrumentation values are unavailable', () => {
  expect(createBabylonPerfAdapter({}, undefined).snapshot()).toEqual({
    drawCalls: 0,
    entities: 0,
    textures: 0,
    visibleTriangles: 0
  })
})
