import { afterEach, describe, expect, it, vi } from 'vitest'

import { setApiRequestConnection, setApiRequestProfile } from '@/hermes'

import { activeConnection, pluginRest } from './plugins'

// A plugin owned by one always-on gateway (news-markets, depot-dashboard) pins
// its REST calls to that registered connection without changing the window's
// active/primary connection. The pin must beat the ambient tag; an absent pin
// must leave the ambient tag intact.
describe('pluginRest connection scope', () => {
  afterEach(() => {
    setApiRequestConnection(null)
    setApiRequestProfile(null)
    Reflect.deleteProperty(window, 'hermesDesktop')
  })

  it('routes one plugin call through an explicitly selected registered connection', async () => {
    const api = vi.fn(async () => ({ outputs: [] }))
    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { api } })
    setApiRequestConnection('local')

    await pluginRest('news-markets', '/outputs', { connectionId: 'hermes-vps' })

    expect(api).toHaveBeenCalledWith(expect.objectContaining({
      connectionId: 'hermes-vps',
      path: '/api/plugins/news-markets/outputs'
    }))
  })

  it('keeps the ambient connection tag when no pin is given', async () => {
    const api = vi.fn(async () => ({}))
    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { api } })
    setApiRequestConnection('some-remote')

    await pluginRest('kanban', '/board')

    expect(api).toHaveBeenCalledWith(expect.objectContaining({ connectionId: 'some-remote' }))
  })
})

// desktop.getConnection/getConnectionFor are IPC round-trips into the main
// process with no timeout of their own (#93454). A wedged main-process
// round-trip must reject instead of hanging pluginSocket's connect() forever.
describe('activeConnection connection timeout (#93454)', () => {
  afterEach(() => {
    setApiRequestConnection(null)
    setApiRequestProfile(null)
    Reflect.deleteProperty(window, 'hermesDesktop')
    vi.useRealTimers()
  })

  it('rejects instead of hanging forever when getConnection() wedges', async () => {
    vi.useFakeTimers()
    setApiRequestProfile('coder')
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { getConnection: vi.fn(() => new Promise(() => undefined)) }
    })

    const pending = expect(activeConnection()).rejects.toThrow('Timed out connecting to profile "coder"')

    await vi.advanceTimersByTimeAsync(20_000)
    await pending
  })

  it('rejects instead of hanging forever when getConnectionFor() wedges', async () => {
    vi.useFakeTimers()
    setApiRequestConnection('gw-tailscale')
    setApiRequestProfile('research')
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: {
        getConnection: vi.fn(() => new Promise(() => undefined)),
        getConnectionFor: vi.fn(() => new Promise(() => undefined))
      }
    })

    const pending = expect(activeConnection()).rejects.toThrow('Timed out connecting to profile "research"')

    await vi.advanceTimersByTimeAsync(20_000)
    await pending
  })
})
