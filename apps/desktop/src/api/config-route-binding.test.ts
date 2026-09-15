import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  getHermesConfigRecord,
  peekConfigReadOrigin,
  resolveConfigWriteScope,
  saveHermesConfig,
  saveHermesConfigRecord,
  setApiRequestConnection,
  setApiRequestProfile
} from '@/hermes'

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

  it('explicit connection/profile pins still win over a captured origin', async () => {
    setApiRequestConnection('connection-a')
    const record = await getHermesConfigRecord()
    setApiRequestConnection('connection-b')

    await saveHermesConfigRecord(record, { connectionId: 'explicit-pin', profile: 'worker' })

    const put = api.mock.calls.find(call => call[0].method === 'PUT')?.[0]

    expect(put).toEqual(
      expect.objectContaining({
        connectionId: 'explicit-pin',
        profile: 'worker'
      })
    )
  })

  it('unbound local writes keep the live ambient path', () => {
    setApiRequestConnection('connection-b')
    setApiRequestProfile('coder')

    expect(resolveConfigWriteScope({ model: 'fresh' })).toEqual({
      connectionId: 'connection-b',
      profile: 'coder'
    })
  })
})
