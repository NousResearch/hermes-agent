import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  getHermesConfigRecord,
  peekConfigReadOrigin,
  retainConfigReadOrigin,
  saveHermesConfig,
  setApiRequestConnection,
  setApiRequestProfile
} from '@/hermes'
import type { HermesConfigRecord } from '@/types/hermes'

describe('config read/write route binding', () => {
  let api: ReturnType<typeof vi.fn>

  beforeEach(() => {
    api = vi.fn(async (request: { method?: string }) =>
      request.method === 'PUT' ? { ok: true } : { model: 'from-read' }
    )
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { api }
    })
    setApiRequestConnection(null)
    setApiRequestProfile(null)
  })

  afterEach(() => {
    setApiRequestConnection(null)
    setApiRequestProfile(null)
    vi.restoreAllMocks()
    Reflect.deleteProperty(window, 'hermesDesktop')
  })

  it('config record read from A cannot be written to B after primary changes', async () => {
    setApiRequestConnection('connection-a')
    setApiRequestProfile('default')

    const record = await getHermesConfigRecord()

    expect(peekConfigReadOrigin(record)).toEqual({ connectionId: 'connection-a', profile: 'default' })
    expect(api).toHaveBeenCalledWith(
      expect.objectContaining({ connectionId: 'connection-a', path: '/api/config', profile: 'default' })
    )

    setApiRequestConnection('connection-b')
    await saveHermesConfig(record)

    const puts = api.mock.calls.filter(call => call[0].method === 'PUT')

    expect(puts).toHaveLength(1)
    expect(puts[0][0]).toEqual(
      expect.objectContaining({
        connectionId: 'connection-a',
        method: 'PUT',
        path: '/api/config',
        profile: 'default'
      })
    )
    expect(puts.filter(call => call[0].connectionId === 'connection-b')).toHaveLength(0)
  })

  it('keeps delayed reads and derived writes isolated for the same profile on two gateways', async () => {
    const readA = Promise.withResolvers<HermesConfigRecord>()
    const readB = Promise.withResolvers<HermesConfigRecord>()
    api.mockReturnValueOnce(readA.promise)
    api.mockReturnValueOnce(readB.promise)

    const ownerA = Object.freeze({ connectionId: 'connection-a', profile: 'worker' })
    const ownerB = Object.freeze({ connectionId: 'connection-b', profile: 'worker' })
    const pendingA = getHermesConfigRecord(ownerA)
    const pendingB = getHermesConfigRecord(ownerB)
    setApiRequestConnection('unrelated-gateway')
    setApiRequestProfile('unrelated-profile')

    readB.resolve({ model: 'model-b' })
    const recordB = await pendingB
    await saveHermesConfig(retainConfigReadOrigin({ ...recordB, model: 'updated-b' }, recordB))
    readA.resolve({ model: 'model-a' })
    const recordA = await pendingA
    await saveHermesConfig(retainConfigReadOrigin({ ...recordA, model: 'updated-a' }, recordA))

    expect(api.mock.calls.map(([request]) => request)).toEqual([
      { ...ownerA, priority: 'foreground', path: '/api/config' },
      { ...ownerB, priority: 'foreground', path: '/api/config' },
      {
        ...ownerB,
        priority: 'foreground',
        path: '/api/config',
        method: 'PUT',
        body: { config: { model: 'updated-b' } }
      },
      {
        ...ownerA,
        priority: 'foreground',
        path: '/api/config',
        method: 'PUT',
        body: { config: { model: 'updated-a' } }
      }
    ])
  })
})
