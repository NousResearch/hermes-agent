import { RECONNECT_STABLE_OPEN_MS } from '@hermes/shared'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { setApiRequestConnection, setApiRequestProfile } from '@/hermes'
import { RECONNECT_ATTEMPT_TIMEOUT_MS } from '@/lib/with-timeout'

import { activeConnection, pluginRest, pluginSocket } from './plugins'

// desktop.getConnection/getConnectionFor are IPC round-trips into the main
// process with no timeout of their own (#93454). A wedged main-process
// round-trip must reject instead of hanging pluginSocket's connect() forever.
describe('activeConnection connection timeout (#93454)', () => {
  it('routes explicit plugin REST scope without borrowing the active connection or profile', async () => {
    setApiRequestConnection('wrong')
    setApiRequestProfile('wrong')
    const api = vi.fn(async () => ({ ok: true }))
    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { api } })
    await pluginRest('demo', '/status', { scope: { connectionId: 'local', profile: 'worker' } })
    expect(api).toHaveBeenCalledWith(
      expect.objectContaining({ path: '/api/plugins/demo/status', connectionId: 'local', profile: 'worker' })
    )
  })
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

// PLG-1: a local gateway Desktop has no static token (authMode 'native'). Its plugin socket must
// dial the one-use URL main mints (the native grant rides a header, never the URL), and a socket
// that keeps being refused must stop instead of re-dialing an idle gateway forever.
describe('pluginSocket credentials and refusals (PLG-1)', () => {
  class FakeSocket {
    static all: FakeSocket[] = []
    readonly at = Date.now()
    closed = false
    onclose: (() => void) | null = null
    onmessage: ((event: { data: unknown }) => void) | null = null
    onopen: (() => void) | null = null
    constructor(readonly url: string) {
      FakeSocket.all.push(this)
    }
    close() {
      this.closed = true
      this.onclose?.()
    }
  }

  const desktop = (connection: Record<string, unknown>, minted: unknown = 'ws://127.0.0.1:9/api/plugins/kanban/events?native_dial=n') => {
    const getPluginWsUrl = vi.fn(async (): Promise<unknown> => minted)
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { getConnection: vi.fn(async () => connection), getPluginWsUrl }
    })

    return getPluginWsUrl
  }

  const local = { authMode: 'native', baseUrl: 'http://127.0.0.1:9', token: '' }

  // Refuse the latest dial (closed before it ever opened) and let the backoff fire.
  const refuseLatest = async () => {
    FakeSocket.all.at(-1)?.onclose?.()
    await vi.advanceTimersByTimeAsync(31_000)
  }

  afterEach(() => {
    FakeSocket.all = []
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
    setApiRequestConnection(null)
    setApiRequestProfile(null)
    Reflect.deleteProperty(window, 'hermesDesktop')
    vi.useRealTimers()
  })

  it('dials the minted local-gateway URL, never an empty ?token=', async () => {
    vi.useFakeTimers()
    vi.stubGlobal('WebSocket', FakeSocket)
    setApiRequestProfile('coder')
    const mint = desktop(local)

    const dispose = pluginSocket('kanban', '/events?board=b', () => undefined)
    await vi.advanceTimersByTimeAsync(0)

    expect(mint).toHaveBeenCalledWith({ connectionId: null, path: '/events?board=b', pluginId: 'kanban', profile: 'coder' })
    expect(FakeSocket.all.map(socket => socket.url)).toEqual(['ws://127.0.0.1:9/api/plugins/kanban/events?native_dial=n'])
    dispose()
  })

  it('stays on polling when main mints no local URL', async () => {
    vi.useFakeTimers()
    vi.stubGlobal('WebSocket', FakeSocket)
    desktop(local, null)

    const dispose = pluginSocket('kanban', '/events', () => undefined)
    await vi.advanceTimersByTimeAsync(120_000)

    expect(FakeSocket.all).toHaveLength(0)
    dispose()
  })

  // The browser cannot tell a refused upgrade from a gateway that is restarting, draining or being
  // re-ensured: those dials keep the capped retry, however long the outage lasts.
  it('keeps retrying through a restart-length outage and reconnects', async () => {
    vi.useFakeTimers()
    vi.stubGlobal('WebSocket', FakeSocket)
    const mint = desktop(local)
    const frames: unknown[] = []

    const dispose = pluginSocket('kanban', '/events', frame => frames.push(frame))
    await vi.advanceTimersByTimeAsync(0)

    for (let i = 0; i < 20; i += 1) {
      await refuseLatest()
    }

    // The mint itself failing while the owner is away is a connection failure too.
    mint.mockRejectedValueOnce(new Error('Gateway ticket control socket missing'))
    await refuseLatest()
    await vi.advanceTimersByTimeAsync(31_000)

    expect(FakeSocket.all).toHaveLength(22)
    expect(mint).toHaveBeenCalledTimes(23)
    const socket = FakeSocket.all.at(-1)!
    socket.onopen?.()
    socket.onmessage?.({ data: '{"ok":1}' })
    expect(frames).toEqual([{ ok: 1 }])
    dispose()
  })

  // A restart can also fail before any dial: the descriptor lookup re-ensures the gateway, which
  // rejects while it starts or drains, or after a stale ticket forgets the cached descriptor.
  const dropOpenSocket = async () => {
    FakeSocket.all.at(-1)?.onopen?.()
    await vi.advanceTimersByTimeAsync(RECONNECT_STABLE_OPEN_MS)
    FakeSocket.all.at(-1)?.close()
  }

  it('keeps retrying while the connection descriptor cannot be resolved, then reconnects', async () => {
    vi.useFakeTimers()
    vi.stubGlobal('WebSocket', FakeSocket)
    desktop(local)
    const getConnection = vi.mocked(window.hermesDesktop.getConnection)
    const frames: unknown[] = []

    const dispose = pluginSocket('kanban', '/events', frame => frames.push(frame))
    await vi.advanceTimersByTimeAsync(0)

    for (let i = 0; i < 8; i += 1) {
      getConnection.mockRejectedValueOnce(new Error('Gateway is draining'))
    }

    await dropOpenSocket()
    await vi.advanceTimersByTimeAsync(9 * 31_000)

    expect(getConnection).toHaveBeenCalledTimes(10)
    expect(FakeSocket.all).toHaveLength(2)
    FakeSocket.all[1].onopen?.()
    FakeSocket.all[1].onmessage?.({ data: '{"ok":1}' })
    expect(frames).toEqual([{ ok: 1 }])
    dispose()
  })

  it('re-dials when the connection descriptor lookup never answers', async () => {
    vi.useFakeTimers()
    vi.stubGlobal('WebSocket', FakeSocket)
    desktop(local)
    const getConnection = vi.mocked(window.hermesDesktop.getConnection)
    const frames: unknown[] = []

    const dispose = pluginSocket('kanban', '/events', frame => frames.push(frame))
    await vi.advanceTimersByTimeAsync(0)
    getConnection.mockImplementationOnce(() => new Promise(() => undefined))
    await dropOpenSocket()
    await vi.advanceTimersByTimeAsync(RECONNECT_ATTEMPT_TIMEOUT_MS + 2 * 31_000)

    expect(getConnection).toHaveBeenCalledTimes(3)
    expect(FakeSocket.all).toHaveLength(2)
    FakeSocket.all[1].onopen?.()
    FakeSocket.all[1].onmessage?.({ data: '{"ok":2}' })
    expect(frames).toEqual([{ ok: 2 }])
    dispose()
  })

  it('a socket waiting on the descriptor dials the new profile once it resolves', async () => {
    vi.useFakeTimers()
    vi.stubGlobal('WebSocket', FakeSocket)
    setApiRequestProfile('coder')
    const mint = desktop(local)
    const getConnection = vi.mocked(window.hermesDesktop.getConnection)

    const dispose = pluginSocket('kanban', '/events', () => undefined)
    await vi.advanceTimersByTimeAsync(0)
    getConnection.mockRejectedValue(new Error('Gateway is starting'))
    await dropOpenSocket()
    await vi.advanceTimersByTimeAsync(600_000)
    expect(FakeSocket.all).toHaveLength(1)

    getConnection.mockResolvedValue(local as never)
    setApiRequestProfile('research')
    await vi.advanceTimersByTimeAsync(31_000)

    expect(getConnection).toHaveBeenLastCalledWith('research')
    expect(mint).toHaveBeenLastCalledWith(expect.objectContaining({ profile: 'research' }))
    expect(FakeSocket.all).toHaveLength(2)
    dispose()
  })

  // Main names the one refusal it can identify: a grant mint refused for a permission reason.
  it('stops after repeated permission refusals from main', async () => {
    vi.useFakeTimers()
    vi.stubGlobal('WebSocket', FakeSocket)
    const mint = desktop(local, { refused: 'permission' })

    const dispose = pluginSocket('kanban', '/events', () => undefined)
    await vi.advanceTimersByTimeAsync(600_000)

    expect(mint).toHaveBeenCalledTimes(5)
    expect(FakeSocket.all).toHaveLength(0)
    dispose()
  })

  it('a profile or connection change resets the refusal count and dials again', async () => {
    vi.useFakeTimers()
    vi.stubGlobal('WebSocket', FakeSocket)
    setApiRequestProfile('coder')
    const mint = desktop(local, { refused: 'permission' })

    const dispose = pluginSocket('kanban', '/events', () => undefined)
    await vi.advanceTimersByTimeAsync(600_000)
    expect(mint).toHaveBeenCalledTimes(5)

    // Still refused under the new profile: a fresh budget of five, then stop again.
    setApiRequestProfile('research')
    await vi.advanceTimersByTimeAsync(600_000)
    expect(mint).toHaveBeenCalledTimes(10)
    expect(mint).toHaveBeenLastCalledWith(expect.objectContaining({ profile: 'research' }))

    mint.mockResolvedValue('ws://127.0.0.1:9/api/plugins/kanban/events?native_dial=m')
    setApiRequestConnection('homelab')
    await vi.advanceTimersByTimeAsync(0)
    expect(mint).toHaveBeenCalledTimes(11)
    expect(FakeSocket.all.map(socket => socket.url)).toEqual(['ws://127.0.0.1:9/api/plugins/kanban/events?native_dial=m'])
    dispose()

    // A disposed socket no longer follows the scope.
    setApiRequestProfile('coder')
    await vi.advanceTimersByTimeAsync(600_000)
    expect(mint).toHaveBeenCalledTimes(11)
  })

  // A grant or descriptor resolved for the profile the user just left, or a socket still bound to
  // it, would feed that profile's plugin stream into the current view.
  const urlFor = (profile: null | string) => `ws://127.0.0.1:9/api/plugins/kanban/events?native_dial=${profile}`

  const profileDesktop = () => {
    const getPluginWsUrl = vi.fn(async ({ profile }: { profile: null | string }): Promise<unknown> => urlFor(profile))
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { getConnection: vi.fn(async () => local), getPluginWsUrl }
    })

    return getPluginWsUrl
  }

  const openUrls = () => FakeSocket.all.filter(socket => !socket.closed).map(socket => socket.url)

  it('a profile switch while the grant is being minted dials the new profile, never the old URL', async () => {
    vi.useFakeTimers()
    vi.stubGlobal('WebSocket', FakeSocket)
    setApiRequestProfile('alpha')
    const mint = profileDesktop()
    let resolveAlpha: (url: unknown) => void = () => undefined
    mint.mockImplementationOnce(() => new Promise(resolve => (resolveAlpha = resolve)))

    const dispose = pluginSocket('kanban', '/events', () => undefined)
    await vi.advanceTimersByTimeAsync(0)
    setApiRequestProfile('beta')
    await vi.advanceTimersByTimeAsync(0)
    resolveAlpha(urlFor('alpha'))
    await vi.advanceTimersByTimeAsync(600_000)

    expect(mint).toHaveBeenLastCalledWith(expect.objectContaining({ profile: 'beta' }))
    expect(FakeSocket.all.map(socket => socket.url)).toEqual([urlFor('beta')])
    dispose()
  })

  it('a profile switch while the connection lookup is pending dials the new profile', async () => {
    vi.useFakeTimers()
    vi.stubGlobal('WebSocket', FakeSocket)
    setApiRequestProfile('alpha')
    const remote = (profile: string) => ({ authMode: 'token', baseUrl: `http://${profile}.invalid`, token: 'tok' })
    let resolveAlpha: (connection: unknown) => void = () => undefined

    const getConnection = vi.fn(async (profile: string): Promise<unknown> => remote(profile))
    getConnection.mockImplementationOnce(() => new Promise(resolve => (resolveAlpha = resolve)))
    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { getConnection } })

    const dispose = pluginSocket('kanban', '/events', () => undefined)
    await vi.advanceTimersByTimeAsync(0)
    setApiRequestProfile('beta')
    await vi.advanceTimersByTimeAsync(0)
    resolveAlpha(remote('alpha'))
    await vi.advanceTimersByTimeAsync(600_000)

    expect(FakeSocket.all.map(socket => socket.url)).toEqual(['ws://beta.invalid/api/plugins/kanban/events?token=tok'])
    dispose()
  })

  it('a profile switch closes the open socket and dials the new profile', async () => {
    vi.useFakeTimers()
    vi.stubGlobal('WebSocket', FakeSocket)
    setApiRequestProfile('alpha')
    const mint = profileDesktop()
    const frames: unknown[] = []

    const dispose = pluginSocket('kanban', '/events', frame => frames.push(frame))
    await vi.advanceTimersByTimeAsync(0)
    const alpha = FakeSocket.all[0]
    alpha.onopen?.()

    setApiRequestProfile('beta')
    await vi.advanceTimersByTimeAsync(0)

    expect(alpha.closed).toBe(true)
    expect(openUrls()).toEqual([urlFor('beta')])
    const beta = FakeSocket.all.at(-1)!
    beta.onopen?.()
    beta.onmessage?.({ data: '{"profile":"beta"}' })
    expect(frames).toEqual([{ profile: 'beta' }])

    // The closed alpha socket neither re-dials nor counts as a refusal.
    await vi.advanceTimersByTimeAsync(600_000)
    expect(FakeSocket.all).toHaveLength(2)
    expect(mint).toHaveBeenCalledTimes(2)
    dispose()
  })

  it('a quick A to B to A switch ends with exactly one open socket, on A', async () => {
    vi.useFakeTimers()
    vi.stubGlobal('WebSocket', FakeSocket)
    setApiRequestProfile('alpha')
    profileDesktop()
    const frames: unknown[] = []

    const dispose = pluginSocket('kanban', '/events', frame => frames.push(frame))
    await vi.advanceTimersByTimeAsync(0)
    FakeSocket.all[0].onopen?.()

    setApiRequestProfile('beta')
    setApiRequestProfile('alpha')
    await vi.advanceTimersByTimeAsync(600_000)

    expect(openUrls()).toEqual([urlFor('alpha')])
    const live = FakeSocket.all.find(socket => !socket.closed)!
    live.onopen?.()
    live.onmessage?.({ data: '{"profile":"alpha"}' })
    expect(frames).toEqual([{ profile: 'alpha' }])
    dispose()
  })

  it('keeps reconnecting a socket that stayed open before it dropped', async () => {
    vi.useFakeTimers()
    vi.stubGlobal('WebSocket', FakeSocket)
    desktop({ authMode: 'token', baseUrl: 'http://remote.invalid', token: 'tok' })

    const dispose = pluginSocket('kanban', '/events', () => undefined)
    await vi.advanceTimersByTimeAsync(0)

    for (let i = 0; i < 10; i += 1) {
      FakeSocket.all.at(-1)?.onopen?.()
      await vi.advanceTimersByTimeAsync(RECONNECT_STABLE_OPEN_MS)
      await refuseLatest()
    }

    expect(FakeSocket.all).toHaveLength(11)
    expect(FakeSocket.all[0].url).toBe('ws://remote.invalid/api/plugins/kanban/events?token=tok')
    dispose()
  })

  // A server that accepts and then closes at once (Kanban on an archived board) must back off
  // and stop like a refusal, not re-dial at the first backoff step forever.
  it('backs off and stops on sockets that open and close at once', async () => {
    vi.useFakeTimers()
    vi.stubGlobal('WebSocket', FakeSocket)
    vi.spyOn(Math, 'random').mockReturnValue(0.5)
    desktop(local)

    const dispose = pluginSocket('kanban', '/events', () => undefined)
    await vi.advanceTimersByTimeAsync(0)
    const delays: number[] = []

    for (let i = 0; i < 20; i += 1) {
      const socket = FakeSocket.all.at(-1)!
      socket.onopen?.()
      const closedAt = Date.now()
      socket.onclose?.()
      await vi.advanceTimersByTimeAsync(31_000)

      if (FakeSocket.all.at(-1) !== socket) {delays.push(FakeSocket.all.at(-1)!.at - closedAt)}
    }

    expect(FakeSocket.all).toHaveLength(5)
    expect(delays).toEqual([250, 500, 1000, 2000])
    dispose()
  })

  // getPluginWsUrl is an IPC round-trip into main with no deadline of its own.
  it('re-dials when the local URL mint never answers', async () => {
    vi.useFakeTimers()
    vi.stubGlobal('WebSocket', FakeSocket)
    const mint = desktop(local)
    mint.mockImplementationOnce(() => new Promise(() => undefined))

    const dispose = pluginSocket('kanban', '/events', () => undefined)
    await vi.advanceTimersByTimeAsync(RECONNECT_ATTEMPT_TIMEOUT_MS + 31_000)

    expect(mint).toHaveBeenCalledTimes(2)
    expect(FakeSocket.all).toHaveLength(1)
    dispose()
  })
})
