import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesConnection } from '@/global'

// pluginSocket grew an onStatus channel so a view can tell
// its live event stream ('live') from the polling fallback ('polling'). The
// status only mirrors the transport — frames prove freshness — and every dead
// end must land on 'polling': no bridge, a mint that resolves null (no ticket
// path for this dial), main refusing the grant five times, a dropped socket,
// and a scope change that closes the old dial. A mint that REJECTS — main's or
// the browser ticket door's — is a connection failure and keeps the capped
// backoff retry. Only a socket that actually opens reports 'live'.

// Direct module imports instead of the '@/hermes' app barrel: the barrel pulls
// the whole UI graph (icons → @tabler/icons-react and friends), which this
// workspace's pruned dependency farm does not carry — and these tests only
// exercise transport logic, no UI.
vi.mock('@/lib/query-client', () => ({ invalidateProfileScopedQueries: vi.fn() }))
vi.mock('@/store/starmap', () => ({ reloadOpenStarmapGraph: vi.fn(), resetStarmapGraph: vi.fn() }))
vi.mock('@/lib/browser-transport', async importOriginal => {
  const actual = await importOriginal<Record<string, unknown>>()

  return {
    ...actual,
    // The browser-hosted Desktop's per-dial ticket mint. Its URL shape and error
    // classification are tested in lib/browser-transport-plugin-events.test.ts and
    // the one-use semantics in the Python ticket tests; here only the outcome.
    pluginEventsTicketUrl: vi.fn()
  }
})

const { pluginSocket } = await import('@/api/plugins')
const { setApiRequestConnection, setApiRequestProfile } = await import('@/api/client')
const { pluginEventsTicketUrl } = await import('@/lib/browser-transport')
const { closeSecondaryGateways, configureGatewayRegistry, setPrimaryGateway } = await import('@/store/gateway')
const { $activeGatewayProfile } = await import('@/store/profile')

const conn = (over: Partial<HermesConnection> = {}): HermesConnection =>
  ({
    authMode: 'oauth',
    baseUrl: 'https://pool.invalid',
    mode: 'remote',
    token: 'fake-test-token',
    wsUrl: 'wss://pool.invalid/api/ws?token=fake-test-token',
    ...over
  }) as HermesConnection

class FakeWebSocket {
  static instances: FakeWebSocket[] = []
  readonly url: string
  onclose: null | (() => void) = null
  onmessage: null | ((event: { data: string }) => void) = null
  onopen: null | (() => void) = null

  constructor(url: string) {
    this.url = url
    FakeWebSocket.instances.push(this)
  }

  close(): void {
    this.onclose?.()
  }
}

let getConnection: ReturnType<typeof vi.fn>

beforeEach(() => {
  FakeWebSocket.instances = []
  vi.stubGlobal('WebSocket', FakeWebSocket)
  getConnection = vi.fn(async (profile?: null | string) => conn({ profile: profile ?? 'default' }))
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: {
      getConnection,
      getGatewayWsUrl: vi.fn(async () => 'wss://pool.invalid/api/ws?ticket=fake')
    }
  })
  configureGatewayRegistry({ onEvent: vi.fn() })
  setPrimaryGateway({ connectionState: 'open' } as never, 'default')
})

afterEach(() => {
  closeSecondaryGateways()
  $activeGatewayProfile.set('default')
  setApiRequestProfile(null)
  setApiRequestConnection(null)
  Reflect.deleteProperty(window, 'hermesDesktop')
  Reflect.deleteProperty(document.documentElement.dataset, 'hermesDesktopHost')
  vi.unstubAllGlobals()
  vi.useRealTimers()
  vi.clearAllMocks()
})

describe('pluginSocket status channel', () => {
  it('reports live when the minted OAuth socket opens and polling when it drops', async () => {
    const getPluginWsUrl = vi.fn(async () => 'wss://pool.invalid/api/plugins/kanban/events?board=ops&ticket=one-use')
    Object.assign(window.hermesDesktop, { getPluginWsUrl })

    const statuses: string[] = []
    const frames: unknown[] = []
    const dispose = pluginSocket('kanban', '/events', data => frames.push(data), status => statuses.push(status))

    await vi.waitFor(() => expect(FakeWebSocket.instances).toHaveLength(1))
    expect(getPluginWsUrl).toHaveBeenCalledWith(expect.objectContaining({ path: '/events', pluginId: 'kanban' }))
    expect(FakeWebSocket.instances[0].url).toContain('ticket=one-use')

    FakeWebSocket.instances[0].onopen?.()
    FakeWebSocket.instances[0].onmessage?.({ data: JSON.stringify({ cursor: 5, events: [] }) })
    FakeWebSocket.instances[0].close()

    expect(frames).toEqual([{ cursor: 5, events: [] }])
    expect(statuses).toEqual(['live', 'polling'])

    dispose()
  })

  it('stays polling without constructing a socket when the mint resolves null', async () => {
    Object.assign(window.hermesDesktop, { getPluginWsUrl: vi.fn(async () => null) })

    const statuses: string[] = []
    const dispose = pluginSocket('kanban', '/events', () => {}, status => statuses.push(status))

    await vi.waitFor(() => expect(statuses).toEqual(['polling']))
    expect(FakeWebSocket.instances).toHaveLength(0)

    dispose()
  })

  it('stays polling at once when the desktop bridge is missing', async () => {
    Reflect.deleteProperty(window, 'hermesDesktop')

    const statuses: string[] = []
    const dispose = pluginSocket('kanban', '/events', () => {}, status => statuses.push(status))

    await vi.waitFor(() => expect(statuses).toEqual(['polling']))
    expect(FakeWebSocket.instances).toHaveLength(0)

    dispose()
  })

  it('stops after five refused mints and leaves the view on polling', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval'] })
    const getPluginWsUrl = vi.fn(async () => ({ refused: 'permission' as const }))
    Object.assign(window.hermesDesktop, { getPluginWsUrl })

    const statuses: string[] = []
    const dispose = pluginSocket('kanban', '/events', () => {}, status => statuses.push(status))

    for (let i = 0; i < 5; i += 1) {
      await vi.advanceTimersByTimeAsync(31_000)
    }

    expect(getPluginWsUrl).toHaveBeenCalledTimes(5)
    expect(statuses).toEqual(['polling'])
    expect(FakeWebSocket.instances).toHaveLength(0)

    dispose()
  })

  it('dials the browser-hosted Desktop through the per-dial ticket mint', async () => {
    document.documentElement.dataset.hermesDesktopHost = 'browser'
    vi.mocked(pluginEventsTicketUrl).mockResolvedValue('wss://web.invalid/api/plugins/kanban/events?board=ops&ticket=b1')

    const statuses: string[] = []
    const dispose = pluginSocket('kanban', '/events', () => {}, status => statuses.push(status))

    await vi.waitFor(() => expect(FakeWebSocket.instances).toHaveLength(1))
    expect(pluginEventsTicketUrl).toHaveBeenCalledTimes(1)
    expect(FakeWebSocket.instances[0].url).toBe('wss://web.invalid/api/plugins/kanban/events?board=ops&ticket=b1')

    FakeWebSocket.instances[0].onopen?.()
    expect(statuses).toEqual(['live'])

    dispose()
  })

  // A rejected browser mint is a connection failure, not the mint's null
  // verdict: backoff must keep re-dialing while the gateway restarts instead of
  // stranding the live channel on polling until the page reloads.
  it('retries a failed browser ticket mint with backoff, never settling on polling', async () => {
    document.documentElement.dataset.hermesDesktopHost = 'browser'
    vi.mocked(pluginEventsTicketUrl).mockRejectedValue(new Error('no session'))

    const statuses: string[] = []
    const dispose = pluginSocket('kanban', '/events', () => {}, status => statuses.push(status))

    await vi.waitFor(() => expect(vi.mocked(pluginEventsTicketUrl).mock.calls.length).toBeGreaterThanOrEqual(3), {
      timeout: 5_000
    })
    expect(FakeWebSocket.instances).toHaveLength(0)
    expect(statuses).toEqual([])

    dispose()
  })

  it('stays polling when the browser mint resolves null — no retry, no socket', async () => {
    document.documentElement.dataset.hermesDesktopHost = 'browser'
    vi.mocked(pluginEventsTicketUrl).mockResolvedValue(null)

    const statuses: string[] = []
    const dispose = pluginSocket('kanban', '/events', () => {}, status => statuses.push(status))

    await vi.waitFor(() => expect(statuses).toEqual(['polling']))
    expect(FakeWebSocket.instances).toHaveLength(0)
    expect(pluginEventsTicketUrl).toHaveBeenCalledTimes(1)

    dispose()
  })

  it('reports polling and redials when the request scope changes', async () => {
    const getPluginWsUrl = vi.fn(async () => `wss://pool.invalid/api/plugins/kanban/events?ticket=${FakeWebSocket.instances.length + 1}`)
    Object.assign(window.hermesDesktop, { getPluginWsUrl })

    const statuses: string[] = []
    const dispose = pluginSocket('kanban', '/events', () => {}, status => statuses.push(status))

    await vi.waitFor(() => expect(FakeWebSocket.instances).toHaveLength(1))
    FakeWebSocket.instances[0].onopen?.()
    expect(statuses).toEqual(['live'])

    setApiRequestProfile('work')

    await vi.waitFor(() => expect(FakeWebSocket.instances).toHaveLength(2))
    expect(getConnection).toHaveBeenLastCalledWith('work')
    // The stale dial's onclose must not speak; the scope listener reported polling.
    FakeWebSocket.instances[0].close()
    expect(statuses).toEqual(['live', 'polling'])

    FakeWebSocket.instances[1].onopen?.()
    expect(statuses).toEqual(['live', 'polling', 'live'])

    dispose()
  })
})
