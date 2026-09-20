import { afterEach, describe, expect, it, vi } from 'vitest'

import { setApiRequestConnection, setApiRequestProfile } from '@/hermes'

import { activeConnection, pluginSocket } from './plugins'

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

describe('pluginSocket authenticated reconnects', () => {
  const connection = {
    authMode: 'oauth' as const,
    baseUrl: 'https://gateway.example/hermes',
    profile: 'research',
    registryScoped: true,
    connectionId: 'cloud',
    token: '',
    wsUrl: 'wss://gateway.example/hermes/api/ws?ticket=stale'
  } as never
  let opened: string[]

  beforeEach(() => {
    opened = []
    vi.useFakeTimers()
    vi.stubGlobal(
      'WebSocket',
      class {
        onclose: null | (() => void) = null
        onmessage: null | ((event: { data: unknown }) => void) = null

        constructor(url: string) {
          opened.push(url)
        }

        close(): void {}
      }
    )
  })

  afterEach(() => {
    setApiRequestConnection(null)
    setApiRequestProfile(null)
    Reflect.deleteProperty(window, 'hermesDesktop')
    vi.useRealTimers()
    vi.unstubAllGlobals()
  })

  it('mints a fresh ticket for the active registry backend before dialing', async () => {
    const getGatewayWsUrlFor = vi.fn().mockResolvedValue({
      ok: true,
      wsUrl: 'wss://gateway.example/hermes/api/ws?ticket=fresh'
    })
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: {
        getConnection: vi.fn(),
        getConnectionFor: vi.fn().mockResolvedValue(connection),
        getGatewayWsUrl: vi.fn(),
        getGatewayWsUrlFor
      }
    })
    setApiRequestConnection('cloud')
    setApiRequestProfile('research')

    const dispose = pluginSocket('kanban', '/events?board=ops', () => {})
    await vi.waitFor(() => expect(opened).toHaveLength(1))

    expect(getGatewayWsUrlFor).toHaveBeenCalledWith({ connectionId: 'cloud', profile: 'research' })
    expect(opened[0]).toBe('wss://gateway.example/hermes/api/plugins/kanban/events?ticket=fresh&board=ops')
    dispose()
  })

  it('retries when the active connection is temporarily unavailable', async () => {
    vi.spyOn(Math, 'random').mockReturnValue(0)
    const getConnection = vi.fn().mockResolvedValueOnce(null).mockResolvedValue({
      ...connection,
      authMode: 'token',
      connectionId: undefined,
      profile: undefined,
      registryScoped: false,
      token: 'long-lived',
      wsUrl: 'wss://gateway.example/hermes/api/ws?token=long-lived'
    })
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { getConnection, getGatewayWsUrl: vi.fn() }
    })

    const dispose = pluginSocket('kanban', '/events', () => {})
    await vi.advanceTimersByTimeAsync(0)
    vi.useRealTimers()
    await vi.waitFor(() => expect(opened).toHaveLength(1))

    expect(getConnection).toHaveBeenCalledTimes(2)
    expect(opened[0]).toBe('wss://gateway.example/hermes/api/plugins/kanban/events?token=long-lived')
    dispose()
  })
})
