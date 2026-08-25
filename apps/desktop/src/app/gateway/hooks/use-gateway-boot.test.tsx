import { act, cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { DesktopConnectionsRegistry } from '@/global'
import { $desktopBoot } from '@/store/boot'
import {
  $connectionsRegistry,
  _resetConnectionsForTests,
  selectConnection,
  setConnectionsRegistry
} from '@/store/connections'
import { closeSecondaryGateways, isActivePrimary } from '@/store/gateway'
import { reconnectGateway } from '@/store/gateway-reconnect'
import { $gatewaySwitching, beginGatewaySwitch, endGatewaySwitch } from '@/store/gateway-switch'
import { notifyError } from '@/store/notifications'
import { $activeGatewayProfile, $profiles, ensureGatewayProfile } from '@/store/profile'
import {
  $activeSessionId,
  $connection,
  $currentCwd,
  $gatewayState,
  $selectedStoredSessionId,
  $sessionsLoading,
  setActiveSessionId,
  setSelectedStoredSessionId
} from '@/store/session'
import { $sessionTiles } from '@/store/session-states'

import { takeGatewaySurvivor } from './gateway-hmr-survivor'
import { primaryRuntimeConnectionId, useGatewayBoot } from './use-gateway-boot'

vi.mock(import('@/store/notifications'), async importOriginal => ({
  ...(await importOriginal()),
  notifyError: vi.fn()
}))

// End-to-end-ish repro of the "remote VPS → stuck on CONNECTING, no Settings"
// bug that drives the REAL useGatewayBoot hook + REAL HermesGateway through a
// fake WebSocket we fully control. No Docker / no real port: from the desktop's
// point of view a "remote VPS" is just a WebSocket that opens once and later
// refuses to reopen, so that is exactly (and only) what we fake.
//
// The previous test (gateway-connecting-overlay.test.tsx) hand-set the stores
// and asserted the overlays; this one proves the HOOK actually PRODUCES that
// stuck store combo — closing the "inferred by reading code" gap on the
// post-boot reconnect loop.

type Listener = (ev: unknown) => void
let connectionApplied: null | (() => void) = null
let powerResume: null | (() => void) = null

describe('primaryRuntimeConnectionId', () => {
  it('uses the registry identity when the primary connection has one', () => {
    expect(primaryRuntimeConnectionId({ connectionId: ' tower ', mode: 'remote' })).toBe('tower')
  })

  it('uses the stable local identity for an app-managed backend', () => {
    expect(primaryRuntimeConnectionId({ mode: 'local' })).toBe('local')
  })

  it('returns null for an unknown remote identity so the caller falls back to live-connection scoping', () => {
    expect(primaryRuntimeConnectionId({ mode: 'remote' })).toBeNull()
  })
})

// Minimal WebSocket stand-in implementing only what json-rpc-gateway.connect()
// touches: readyState, add/removeEventListener('open'|'error'|'close'), close().
class FakeWebSocket {
  static OPEN = 1
  static CLOSED = 3
  // Flipped by the test: 'open' = next socket connects; 'fail' = next socket
  // errors (a dead remote). Mirrors a VPS going away after the first connect.
  static mode: 'open' | 'fail' = 'open'
  static instances: FakeWebSocket[] = []
  // Ping behavior: 'pong' answers with a healthy pong frame; 'silent' swallows
  // the request (the half-open-socket simulation — connection looks OPEN but
  // every RPC hangs until its per-call timeout); 'method-not-found' answers
  // the JSON-RPC error a PRE-ping backend returns (a healthy, version-skewed
  // response that must NOT trigger a reconnect).
  static pingMode: 'pong' | 'silent' | 'method-not-found' = 'pong'

  readyState = 0
  private listeners: Record<string, Set<Listener>> = {}

  constructor(public url: string) {
    FakeWebSocket.instances.push(this)
    const willOpen = FakeWebSocket.mode === 'open'
    // Resolve on the next microtask/macrotask so connect()'s promise wiring is
    // in place before open/error fires (matches real async socket handshake).
    setTimeout(() => {
      if (willOpen) {
        this.readyState = FakeWebSocket.OPEN
        this.emit('open', {})
      } else {
        this.readyState = FakeWebSocket.CLOSED
        this.emit('error', {})
      }
    }, 0)
  }

  addEventListener(type: string, fn: Listener) {
    ;(this.listeners[type] ??= new Set()).add(fn)
  }

  removeEventListener(type: string, fn: Listener) {
    this.listeners[type]?.delete(fn)
  }

  close() {
    this.readyState = FakeWebSocket.CLOSED
    this.emit('close', {})
  }

  // Force-drop an open socket, as a sleeping laptop / restarted remote would.
  drop() {
    this.readyState = FakeWebSocket.CLOSED
    this.emit('close', {})
  }

  send(data: string) {
    let frame: { id?: unknown; method?: string }

    try {
      frame = JSON.parse(data) as { id?: unknown; method?: string }
    } catch {
      return
    }

    if (frame.method !== 'ping') {
      return
    }

    if (FakeWebSocket.pingMode === 'pong') {
      this.emit('message', {
        data: JSON.stringify({ jsonrpc: '2.0', id: frame.id, result: { pong: true } })
      })
    } else if (FakeWebSocket.pingMode === 'method-not-found') {
      this.emit('message', {
        data: JSON.stringify({
          jsonrpc: '2.0',
          id: frame.id,
          error: { code: -32601, message: 'Method not found' }
        })
      })
    }
    // 'silent': swallow — a healthy socket answers, a half-open one never does.
  }

  private emit(type: string, ev: unknown) {
    for (const fn of this.listeners[type] ?? []) {
      fn(ev)
    }
  }
}

const primaryConn = {
  authMode: 'token' as const,
  baseUrl: 'https://vps.example.com',
  connectionId: 'primary-vps',
  profile: 'default',
  token: 't',
  wsUrl: 'wss://vps.example.com/api/ws?token=t'
}

const coderConn = {
  authMode: 'token' as const,
  baseUrl: 'https://coder.example.com',
  connectionId: 'coder-remote',
  profile: 'coder',
  token: 'c',
  wsUrl: 'wss://coder.example.com/api/ws?token=c'
}

function fakeDesktop() {
  let bootProgressHandler: ((payload: Record<string, unknown>) => void) | null = null

  return {
    getConnection: vi.fn(async (profile?: null | string) => {
      const key = (profile ?? '').trim()

      return !key || key === 'default' ? primaryConn : coderConn
    }),
    getGatewayWsUrl: vi.fn(async (conn?: { wsUrl?: string }) => conn?.wsUrl ?? primaryConn.wsUrl),
    getBootProgress: vi.fn(async () => ({
      error: null as null | string,
      fakeMode: false,
      message: '',
      phase: 'init',
      progress: 0,
      retryable: false as boolean,
      running: true as boolean,
      timestamp: Date.now()
    })),
    onBootProgress: vi.fn(callback => {
      bootProgressHandler = callback

      return () => {
        bootProgressHandler = null
      }
    }),
    // Test helper: fire a post-boot progress event through the real subscription.
    emitBootProgress(payload: Record<string, unknown>) {
      bootProgressHandler?.(payload)
    },
    onBackendExit: vi.fn(() => () => undefined),
    onConnectionApplied: vi.fn(callback => {
      connectionApplied = callback

      return () => {
        connectionApplied = null
      }
    }),
    onPowerResume: vi.fn(callback => {
      powerResume = callback

      return () => {
        powerResume = null
      }
    }),
    revalidateConnection: vi.fn(async () => ({ ok: true, rebuilt: false })),
    onWindowStateChanged: vi.fn(() => () => undefined),
    touchBackend: vi.fn(async () => undefined),
    profile: { get: vi.fn(async () => ({ profile: 'default' })) }
  }
}

function Harness({
  beforeConnectionSwitch = () => undefined,
  refreshSessions
}: { beforeConnectionSwitch?: () => void; refreshSessions?: () => Promise<void> } = {}) {
  useGatewayBoot({
    beforeConnectionSwitch,
    handleGatewayEvent: () => undefined,
    onConnectionReady: () => undefined,
    onGatewayReady: () => undefined,
    refreshHermesConfig: async () => undefined,
    refreshSessions: refreshSessions ?? (async () => undefined)
  })

  return null
}

const originalWebSocket = globalThis.WebSocket

beforeEach(() => {
  // Drop any parked gateway left by a prior file/case (globalThis slot).
  const leftover = takeGatewaySurvivor()

  if (leftover) {
    try {
      leftover.gateway.close()
    } catch {
      // ignore
    }
  }

  closeSecondaryGateways()
  $activeGatewayProfile.set('default')
  $connection.set(null)
  $profiles.set([])
  $sessionTiles.set([])
  vi.useFakeTimers()
  FakeWebSocket.mode = 'open'
  FakeWebSocket.instances = []
  FakeWebSocket.pingMode = 'pong'
  connectionApplied = null
  powerResume = null
  vi.mocked(notifyError).mockReset()
  ;(globalThis as { WebSocket: unknown }).WebSocket = FakeWebSocket
  ;(window as { hermesDesktop?: unknown }).hermesDesktop = fakeDesktop()
  $gatewayState.set('idle')
  $desktopBoot.set({
    error: null,
    fakeMode: false,
    message: '',
    phase: 'init',
    progress: 0,
    running: true,
    timestamp: Date.now(),
    visible: true
  })
})

afterEach(() => {
  cleanup()
  // Vitest keeps import.meta.hot truthy, so the boot effect's cleanup parks an
  // open gateway instead of tearing it down (the real HMR path). Drain + close
  // that survivor so the next test boots a fresh socket instead of adoptBoot().
  const survivor = takeGatewaySurvivor()

  if (survivor) {
    try {
      survivor.gateway.close()
    } catch {
      // ignore
    }
  }

  closeSecondaryGateways()
  $activeGatewayProfile.set('default')
  $connection.set(null)
  $profiles.set([])
  $sessionTiles.set([])
  _resetConnectionsForTests()
  $connectionsRegistry.set(null)
  setActiveSessionId(null)
  setSelectedStoredSessionId(null)
  endGatewaySwitch()
  vi.useRealTimers()
  ;(globalThis as { WebSocket: unknown }).WebSocket = originalWebSocket
  delete (window as { hermesDesktop?: unknown }).hermesDesktop
  window.localStorage.removeItem('hermes.desktop.workspace-cwd')
  $currentCwd.set('')
})

// Let pending microtasks (awaits) AND the queued 0ms socket open/error fire.
async function flushAsync() {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(0)
  })
}

// Drive the exponential backoff forward by its full cap so the next scheduled
// reconnect attempt actually runs (1s,2s,4s,8s,15s,15s…). Returns after the
// attempt's async work settles.
async function advanceBackoff() {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(15_000)
  })
}

describe('useGatewayBoot remote reconnect loop (real hook, fake socket)', () => {
  it('INITIAL boot against a dead VPS: getConnection hangs (waitForHermes) → app sits in the connecting combo, then fails', async () => {
    // The report's actual path: a fresh launch pointed at an unreachable VPS.
    // startHermes()'s remote branch awaits waitForHermes() for 45s before it
    // throws, so the renderer's `await desktop.getConnection()` stays pending
    // that whole window. During it: gatewayState is still 'idle' (connect was
    // never reached) and boot.error is null → connecting=true → the fullscreen
    // CONNECTING overlay, latched, blocking Settings.
    let rejectConn: (e: Error) => void = () => undefined
    const desktop = fakeDesktop()
    desktop.getConnection = vi.fn(
      () =>
        new Promise((_resolve, reject) => {
          rejectConn = reject
        })
    )
    ;(window as { hermesDesktop?: unknown }).hermesDesktop = desktop

    render(<Harness />)
    await flushAsync()

    // getConnection is still pending — the dead-VPS wait. No socket was ever
    // created, gatewayState never left idle, boot.error is null.
    expect(FakeWebSocket.instances).toHaveLength(0)
    expect($gatewayState.get()).not.toBe('open')
    expect($desktopBoot.get().error).toBeNull()
    // ^ connecting === true here → fullscreen CONNECTING, no Settings.

    // After ~45s waitForHermes gives up and getConnection rejects → boot()
    // catch → failDesktopBoot → the BootFailureOverlay recovery surface.
    await act(async () => {
      rejectConn(new Error('Hermes backend did not become ready: timeout'))
      await vi.advanceTimersByTimeAsync(0)
    })

    expect($desktopBoot.get().error).toBeTruthy()
  })

  it('resets the old machine context before connecting an applied gateway', async () => {
    const beforeConnectionSwitch = vi.fn()
    render(<Harness beforeConnectionSwitch={beforeConnectionSwitch} />)
    await flushAsync()
    expect(connectionApplied).not.toBeNull()

    act(() => connectionApplied?.())
    expect(beforeConnectionSwitch).toHaveBeenCalledTimes(1)
    await flushAsync()
    expect($gatewayState.get()).toBe('open')
  })

  it('a stale failed Settings switch cannot disarm the newer switch loading owner', async () => {
    const desktop = fakeDesktop()

    ;(window as { hermesDesktop?: unknown }).hermesDesktop = desktop

    render(<Harness />)
    await flushAsync()
    expect($gatewayState.get()).toBe('open')

    const failureA = new Error('switch A failed')
    const failureB = new Error('switch B failed')
    let rejectA: (error: Error) => void = () => undefined
    let rejectB: (error: Error) => void = () => undefined

    desktop.getConnection
      .mockImplementationOnce(
        () =>
          new Promise((_resolve, reject) => {
            rejectA = reject
          })
      )
      .mockImplementationOnce(
        () =>
          new Promise((_resolve, reject) => {
            rejectB = reject
          })
      )

    act(() => connectionApplied?.())
    expect($gatewaySwitching.get()).toBe(true)
    expect($sessionsLoading.get()).toBe(true)

    act(() => connectionApplied?.())
    expect($gatewaySwitching.get()).toBe(true)
    expect($sessionsLoading.get()).toBe(true)

    await act(async () => {
      rejectA(failureA)
      await vi.advanceTimersByTimeAsync(0)
    })

    expect(notifyError).toHaveBeenCalledWith(failureA, expect.any(String))
    expect($desktopBoot.get().error).toBe(failureA.message)
    expect($gatewaySwitching.get()).toBe(true)
    expect($sessionsLoading.get()).toBe(true)

    await act(async () => {
      rejectB(failureB)
      await vi.advanceTimersByTimeAsync(0)
    })

    expect(notifyError).toHaveBeenCalledWith(failureB, expect.any(String))
    expect($desktopBoot.get().error).toBe(failureB.message)
    expect($gatewaySwitching.get()).toBe(false)
    expect($sessionsLoading.get()).toBe(false)
  })

  it('reports a Settings switch setup failure and does not disarm a newer switch started by recovery UI', async () => {
    const failure = new Error('machine-context reset failed')
    const beforeConnectionSwitch = vi.fn()
    let newerToken: null | ReturnType<typeof beginGatewaySwitch> = null

    beforeConnectionSwitch.mockImplementationOnce(() => {
      throw failure
    })
    vi.mocked(notifyError).mockImplementationOnce((_error, fallback) => {
      // A notification/recovery callback may synchronously start another
      // switch. The failed Settings attempt never received a token and must
      // not force this newer owner's barrier down from its finally block.
      newerToken = beginGatewaySwitch()

      return fallback
    })

    render(<Harness beforeConnectionSwitch={beforeConnectionSwitch} />)
    await flushAsync()
    expect($gatewayState.get()).toBe('open')

    act(() => connectionApplied?.())
    await flushAsync()

    expect($desktopBoot.get().error).toBe(failure.message)
    expect(notifyError).toHaveBeenCalledWith(failure, expect.any(String))
    expect(newerToken).not.toBeNull()
    expect($gatewaySwitching.get()).toBe(true)
    expect($sessionsLoading.get()).toBe(true)

    endGatewaySwitch(newerToken ?? undefined)
  })

  it('a store-driven switch (Sessions switcher) runs the same machine-context reset as a Settings apply (#93937)', async () => {
    const beforeConnectionSwitch = vi.fn()
    const { unmount } = render(<Harness beforeConnectionSwitch={beforeConnectionSwitch} />)
    await flushAsync()

    act(() => beginGatewaySwitch())
    expect(beforeConnectionSwitch).toHaveBeenCalledTimes(1)
    expect($gatewaySwitching.get()).toBe(true)
    act(() => endGatewaySwitch())
    expect($gatewaySwitching.get()).toBe(false)

    // Teardown unregisters: a switch after unmount must not call a dead host.
    unmount()
    beginGatewaySwitch()
    endGatewaySwitch()
    expect(beforeConnectionSwitch).toHaveBeenCalledTimes(1)
  })

  it("#93937: the Sessions switcher never publishes the new source while the previous backend's runtime id is still bound", async () => {
    // Real stores end to end: real useGatewayBoot, real gateway registry, real
    // selectConnection, fake sockets. Boot on the primary VPS with a transcript
    // open (its runtime id was minted by THAT backend), then switch sources
    // through the sidebar door. Before the fix that door activated the new
    // socket first and wiped the bindings after an IPC round-trip, so the
    // renderer sat on "gateway B + runtime id from A" and B answered every
    // session RPC with "session not found".
    const registryConnections: DesktopConnectionsRegistry = {
      connections: [
        { id: 'primary-vps', kind: 'remote', label: 'VPS', tokenPreview: '...t', tokenSet: true },
        { id: 'coder-remote', kind: 'remote', label: 'Coder', tokenPreview: '...c', tokenSet: true }
      ],
      primary: 'primary-vps',
      secureTokenStorage: true,
      version: 2
    }

    const desktop = fakeDesktop() as ReturnType<typeof fakeDesktop> & Record<string, unknown>
    const setLastUsed = vi.fn(async (id: string) => ({ ok: true, registry: { ...registryConnections, lastUsed: id } }))
    let bindingAtDial: null | string = null

    desktop.api = vi.fn(async ({ path }: { path: string }) =>
      path === '/api/profiles/active' ? { active: 'default', current: 'default' } : { profiles: [] }
    )
    desktop.getConnectionFor = vi.fn(async ({ connectionId, profile }: { connectionId: string; profile: string }) => ({
      ...coderConn,
      connectionId,
      profile,
      registryScoped: true
    }))
    desktop.getGatewayWsUrlFor = vi.fn(async () => {
      // Phase 1 (the dial) runs with the previous source still fully bound.
      bindingAtDial = $activeSessionId.get()

      return coderConn.wsUrl
    })
    desktop.connections = { list: vi.fn(async () => registryConnections), setLastUsed }
    ;(window as { hermesDesktop?: unknown }).hermesDesktop = desktop

    const beforeConnectionSwitch = vi.fn()
    render(<Harness beforeConnectionSwitch={beforeConnectionSwitch} />)
    await flushAsync()
    expect($gatewayState.get()).toBe('open')
    expect($connection.get()?.connectionId).toBe('primary-vps')

    setConnectionsRegistry(registryConnections)
    setSelectedStoredSessionId('stored-on-vps')
    setActiveSessionId('a93bb39d')

    // Every instant the new source is visible, with what a session-scoped
    // effect would read right then.
    const published: Array<{ activeSessionId: null | string; switching: boolean }> = []

    const off = $connection.listen(next => {
      if (next?.connectionId === 'coder-remote') {
        published.push({ activeSessionId: $activeSessionId.get(), switching: $gatewaySwitching.get() })
      }
    })

    const switching = selectConnection('coder-remote')
    await flushAsync()
    await flushAsync()
    await flushAsync()
    await switching
    off()

    // The previous backend's runtime id was already gone — and the barrier up —
    // at every publication of the new source. (Pre-fix: the first publication
    // carried activeSessionId 'a93bb39d' with the barrier down.)
    expect(published.length).toBeGreaterThan(0)
    expect(published).toEqual(published.map(() => ({ activeSessionId: null, switching: true })))
    expect(bindingAtDial).toBe('a93bb39d')
    expect(beforeConnectionSwitch).toHaveBeenCalledTimes(1)
    expect($connection.get()?.connectionId).toBe('coder-remote')
    expect(isActivePrimary()).toBe(false)
    expect($activeSessionId.get()).toBeNull()
    expect($selectedStoredSessionId.get()).toBeNull()
    expect($gatewaySwitching.get()).toBe(false)
    // The switch committed: the registry remembers the new source as last-used.
    expect(setLastUsed).toHaveBeenCalledWith('coder-remote')
  })

  it('re-fetches the profile rail from the NEW backend after a connection apply (#85731)', async () => {
    // The reported repro: connected to backend A, the rail shows A's named
    // profiles; the user applies a different remote/Cloud connection (soft
    // re-home). The rail must repopulate from backend B — before the fix
    // nothing deterministically re-pulled /api/profiles on the soft switch,
    // so the rail kept (or, with a stale in-flight response, collapsed to)
    // the previous backend's list.
    const desktop = fakeDesktop() as ReturnType<typeof fakeDesktop> & {
      api: ReturnType<typeof vi.fn>
    }

    desktop.api = vi.fn(async ({ path }: { path: string }) => {
      if (path === '/api/profiles/active') {
        return { active: 'default', current: 'default' }
      }

      if (path === '/api/profiles') {
        return {
          profiles: [
            { is_default: true, name: 'default' },
            { is_default: false, name: 'cloud-eric' }
          ]
        }
      }

      throw new Error(`unexpected api call: ${path}`)
    })
    ;(window as { hermesDesktop?: unknown }).hermesDesktop = desktop

    render(<Harness />)
    await flushAsync()
    expect($gatewayState.get()).toBe('open')

    // The rail currently mirrors backend A's profile universe.
    $profiles.set([
      { is_default: true, name: 'default' },
      { is_default: false, name: 'eric' }
    ] as never)

    // Settings → Gateway apply lands: main tears down softly and notifies.
    act(() => connectionApplied?.())
    await flushAsync()
    await flushAsync()

    expect($gatewayState.get()).toBe('open')
    // Backend B's list replaced A's — the rail survives the switch instead of
    // painting the previous backend's (or an empty) universe.
    expect($profiles.get().map(profile => profile.name)).toEqual(['default', 'cloud-eric'])
  })

  it('a remote that drops post-boot keeps looping with NO boot.error (the dead-end CONNECTING combo)', async () => {
    render(<Harness />)
    await flushAsync()

    // Initial boot connected.
    expect($gatewayState.get()).toBe('open')
    expect($desktopBoot.get().error).toBeNull()
    expect(FakeWebSocket.instances).toHaveLength(1)

    // The remote VPS goes away: drop the live socket, and make every reopen
    // fail from here on.
    FakeWebSocket.mode = 'fail'
    act(() => FakeWebSocket.instances[0].drop())
    await flushAsync()

    // Burn a couple backoff cycles BEFORE the escalation threshold. Socket
    // down, hook retrying, gatewayState non-open, boot.error still null so
    // chat stays usable (no CONNECTING / no couldn't-start overlay).
    await advanceBackoff()

    expect($gatewayState.get()).not.toBe('open')
    expect($desktopBoot.get().error).toBeNull()
    // It is actively retrying, not idle — more sockets were minted.
    expect(FakeWebSocket.instances.length).toBeGreaterThan(1)
  })

  it('FIX: after a prolonged drop the chat stays unlocked (toast, not boot.error)', async () => {
    render(<Harness />)
    await flushAsync()
    expect($desktopBoot.get().error).toBeNull()

    FakeWebSocket.mode = 'fail'
    act(() => FakeWebSocket.instances[0].drop())
    await flushAsync()

    // Walk the backoff well past the historical 45s threshold and into the
    // current multi-minute escalate window. Chat must stay unlocked either way.
    for (let i = 0; i < 24; i += 1) {
      await advanceBackoff()
    }

    // Transport blips must NOT take the full-screen "couldn't start" overlay —
    // users were locked out of reading/drafting for the whole reconnect window.
    expect($desktopBoot.get().error).toBeNull()
    expect($gatewayState.get()).not.toBe('open')
    // Still retrying.
    expect(FakeWebSocket.instances.length).toBeGreaterThan(1)
  })

  it('FIX: a successful reconnect after a prolonged drop restores the open gateway', async () => {
    render(<Harness />)
    await flushAsync()

    FakeWebSocket.mode = 'fail'
    act(() => FakeWebSocket.instances[0].drop())
    await flushAsync()

    for (let i = 0; i < 24; i += 1) {
      await advanceBackoff()
    }

    expect($desktopBoot.get().error).toBeNull()

    // The remote comes back: next reconnect attempt opens.
    FakeWebSocket.mode = 'open'
    await advanceBackoff()

    expect($gatewayState.get()).toBe('open')
    expect($desktopBoot.get().error).toBeNull()
  })

  it('a getConnection() that hangs on reconnect does not permanently latch the backoff loop (#93454)', async () => {
    // Repro: a remote gateway drops, the backoff loop kicks off a reconnect,
    // and the IPC round-trip into main (desktop.getConnection) never settles
    // — e.g. a wedged revalidation after a liveness-probe trip, even though
    // the backend itself answers fine. Without an internal timeout on that
    // await, `reconnecting` never clears and every later
    // scheduleReconnect()/attemptReconnect() early-returns forever, so the UI
    // stays stuck until the app is restarted.
    const desktop = fakeDesktop()
    const originalGetConnection = desktop.getConnection
    let callCount = 0

    desktop.getConnection = vi.fn((profile?: null | string) => {
      callCount += 1

      // The initial boot call succeeds; every reconnect attempt after the
      // drop hangs indefinitely.
      return callCount === 1 ? originalGetConnection(profile) : new Promise(() => undefined)
    })
    ;(window as { hermesDesktop?: unknown }).hermesDesktop = desktop

    render(<Harness />)
    await flushAsync()
    expect($gatewayState.get()).toBe('open')
    expect(callCount).toBe(1)

    act(() => FakeWebSocket.instances[0].drop())
    await advanceBackoff()

    expect(callCount).toBe(2)
    expect($gatewayState.get()).not.toBe('open')

    // Advance past the internal reconnect-attempt timeout (20s) — the stalled
    // await must reject so the `reconnecting` guard clears and the backoff
    // loop schedules another attempt, instead of latching forever on the
    // still-pending first hang.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(20_000)
    })
    await advanceBackoff()

    expect(callCount).toBeGreaterThanOrEqual(3)
  })

  it('a revalidateConnection() that hangs on reconnect does not permanently latch the backoff loop (#93454)', async () => {
    // Same failure mode as the getConnection() repro above, but for the OTHER
    // unbounded IPC await in the same try block: a wedged revalidation after a
    // liveness-probe trip (the PR's own named trigger) must also unlatch.
    const desktop = fakeDesktop()
    let revalidateCallCount = 0

    desktop.revalidateConnection = vi.fn(() => {
      revalidateCallCount += 1

      // Every reconnect attempt after the drop hangs indefinitely; getConnection
      // itself stays fast so this isolates the revalidate call specifically.
      return new Promise(() => undefined)
    })
    ;(window as { hermesDesktop?: unknown }).hermesDesktop = desktop

    render(<Harness />)
    await flushAsync()
    expect($gatewayState.get()).toBe('open')
    const callsBeforeDrop = desktop.getConnection.mock.calls.length

    act(() => FakeWebSocket.instances[0].drop())
    await advanceBackoff()

    expect(revalidateCallCount).toBe(1)
    expect($gatewayState.get()).not.toBe('open')
    // Still stuck behind the hung revalidate — execution never reached
    // getConnection() at all.
    expect(desktop.getConnection.mock.calls.length).toBe(callsBeforeDrop)

    // Advance past the internal reconnect-attempt timeout (20s) — the stalled
    // revalidate await must reject (swallowed, as it always was for a genuine
    // rejection) so execution proceeds to getConnection() and the socket
    // reopens, instead of latching on the still-pending revalidate forever.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(20_000)
    })

    expect(desktop.getConnection.mock.calls.length).toBeGreaterThan(callsBeforeDrop)
    expect($gatewayState.get()).toBe('open')
  })

  it('rebinds Bot tabs owned by the restarted primary without touching another gateway', async () => {
    render(<Harness />)
    await flushAsync()
    $sessionTiles.set([
      {
        ownerRoute: { connectionId: 'primary-vps', mode: 'remote', profile: 'writer', targetProfile: 'writer' },
        runtimeId: 'runtime-primary-dead',
        storedSessionId: 'primary-bot-chat',
        workspaceMode: 'bots',
        workspaceOwnerKey: 'primary-vps::writer'
      },
      {
        ownerRoute: { connectionId: 'coder-remote', mode: 'remote', profile: 'coder', targetProfile: 'coder' },
        runtimeId: 'runtime-secondary-live',
        storedSessionId: 'secondary-bot-chat',
        workspaceMode: 'bots',
        workspaceOwnerKey: 'coder-remote::coder'
      }
    ])

    act(() => FakeWebSocket.instances[0].drop())
    FakeWebSocket.mode = 'open'
    await advanceBackoff()

    const [primaryBot, secondaryBot] = $sessionTiles.get()

    expect(primaryBot).not.toHaveProperty('runtimeId')
    expect(secondaryBot).toMatchObject({ runtimeId: 'runtime-secondary-live' })
  })

  it('manual reconnect revalidates, re-resolves, re-mints, and re-dials the dropped socket', async () => {
    const desktop = fakeDesktop()

    ;(window as { hermesDesktop?: unknown }).hermesDesktop = desktop

    render(<Harness />)
    await flushAsync()

    expect($gatewayState.get()).toBe('open')
    act(() => FakeWebSocket.instances[0].drop())
    FakeWebSocket.mode = 'open'

    await act(async () => {
      const reconnect = reconnectGateway()
      await vi.advanceTimersByTimeAsync(0)
      await reconnect
    })

    expect(desktop.revalidateConnection).toHaveBeenCalledOnce()
    // The manual reconnect dials the WINDOW-owned primary backend (no profile
    // arg) — same contract as the sleep/wake reconnect: passing the active
    // profile would retarget the primary socket after a live profile swap.
    const lastCall = desktop.getConnection.mock.calls.at(-1) ?? []
    expect(lastCall.length === 0 || lastCall[0] == null || lastCall[0] === '').toBe(true)
    expect(desktop.getGatewayWsUrl).toHaveBeenCalledTimes(2)
    expect(FakeWebSocket.instances).toHaveLength(2)
    expect($gatewayState.get()).toBe('open')
  })

  it('power resume force-redials a half-open primary socket that still reports OPEN', async () => {
    const desktop = fakeDesktop()

    ;(window as { hermesDesktop?: unknown }).hermesDesktop = desktop

    render(<Harness />)
    await flushAsync()

    const staleSocket = FakeWebSocket.instances[0]

    expect(staleSocket.readyState).toBe(FakeWebSocket.OPEN)
    expect($gatewayState.get()).toBe('open')
    expect(powerResume).not.toBeNull()

    // macOS can discard the TCP connection during sleep without updating the
    // renderer WebSocket object. Leave readyState OPEN, swallow the liveness
    // ping (a half-open socket never answers), and emit only resume. The wake
    // path no longer blind-closes an open-looking socket — it probes first and
    // closes only when the probe times out.
    FakeWebSocket.pingMode = 'silent'
    act(() => powerResume?.())
    await act(async () => {
      await vi.advanceTimersByTimeAsync(5_100)
    })

    expect(staleSocket.readyState).toBe(FakeWebSocket.CLOSED)
    // The probe-driven close schedules the regular backoff reconnect, which
    // revalidates the (possibly dead) remote descriptor before re-dialing.
    await advanceBackoff()
    expect(desktop.revalidateConnection).toHaveBeenCalledOnce()
    expect(FakeWebSocket.instances).toHaveLength(2)
    expect($gatewayState.get()).toBe('open')
  })

  it('FIX: post-boot ticket-mint boot-progress errors do not lock the UI', async () => {
    const desktop = fakeDesktop() as ReturnType<typeof fakeDesktop> & {
      emitBootProgress: (payload: Record<string, unknown>) => void
    }

    ;(window as { hermesDesktop?: unknown }).hermesDesktop = desktop

    render(<Harness />)
    await flushAsync()
    expect($gatewayState.get()).toBe('open')
    expect($desktopBoot.get().error).toBeNull()

    // Main re-emits the exact transient ticket error after a liveness rebuild.
    // That used to promote into BootFailureOverlay and lock reading/drafting.
    act(() => {
      desktop.emitBootProgress({
        error: 'Could not reach the remote Hermes gateway while refreshing its WebSocket ticket. Try reconnecting.',
        message: 'Desktop boot failed',
        phase: 'backend.error',
        progress: 94,
        running: false,
        timestamp: Date.now()
      })
    })

    expect($desktopBoot.get().error).toBeNull()
    expect($desktopBoot.get().visible).toBe(false)
  })

  it('FIX: a failed session-list fetch during boot is non-fatal — the app still boots', async () => {
    // The version-skew report: gateway WS connects fine, but refreshSessions()
    // rejects (e.g. older backend 404s an endpoint the fallback didn't cover,
    // or a transient read error). That must NOT reject boot() into
    // failDesktopBoot's "Hermes couldn't start" overlay — the socket is open
    // and the app is fully usable with an empty sidebar.
    const refreshSessions = vi.fn(async () => {
      throw new Error('404: {"detail":"No such API endpoint: /api/profiles/sessions/sidebar"}')
    })

    render(<Harness refreshSessions={refreshSessions} />)
    await flushAsync()

    expect(refreshSessions).toHaveBeenCalled()
    expect($gatewayState.get()).toBe('open')
    // Boot completed: no error, overlay dismissed.
    expect($desktopBoot.get().error).toBeNull()
    expect($desktopBoot.get().visible).toBe(false)
    expect($desktopBoot.get().phase).toBe('renderer.ready')
  })

  it('seeds the configured default project dir pre-connect — no route-resume race (#71873)', async () => {
    // The reporter's scenario: a configured default project dir must be applied
    // at boot regardless of route-resume timing. The seed now runs BEFORE the
    // gateway opens, so no session restore can race it (route-resume is gated
    // on gatewayState === 'open').
    const desktop = fakeDesktop() as {
      sanitizeWorkspaceCwd?: unknown
      settings?: unknown
    }

    desktop.settings = {
      getDefaultProjectDir: vi.fn(async () => ({
        defaultLabel: 'C:\\Users\\sonny',
        dir: 'C:\\Hermes',
        resolvedCwd: 'C:\\Hermes'
      })),
      pickDefaultProjectDir: vi.fn(async () => undefined),
      setDefaultProjectDir: vi.fn(async () => undefined)
    }
    desktop.sanitizeWorkspaceCwd = vi.fn(async (cwd: string) => ({ cwd }))
    ;(window as { hermesDesktop?: unknown }).hermesDesktop = desktop

    // Record the cwd at the exact moment the gateway opens its WebSocket: if
    // the seed moved back post-connect, this would still be '' here and the
    // end-state assertion would pass anyway (the seed would run later in the
    // same flush). The construction-time snapshot is what proves ordering.
    let cwdAtConnect = ''

    class RecordingSocket extends FakeWebSocket {
      constructor(url: string) {
        super(url)
        cwdAtConnect = $currentCwd.get()
      }
    }

    ;(globalThis as { WebSocket: unknown }).WebSocket = RecordingSocket

    render(<Harness />)
    await flushAsync()

    expect(cwdAtConnect).toBe('C:\\Hermes')
    expect($currentCwd.get()).toBe('C:\\Hermes')
  })

  it('FIX: primary sleep/wake reconnect dials the window backend, not the active secondary profile', async () => {
    const desktop = fakeDesktop()

    ;(window as { hermesDesktop?: unknown }).hermesDesktop = desktop

    render(<Harness />)
    await flushAsync()
    expect($gatewayState.get()).toBe('open')
    expect(FakeWebSocket.instances).toHaveLength(1)
    expect(FakeWebSocket.instances[0].url).toBe(primaryConn.wsUrl)

    // Profile swap opens a secondary WS; briefly use real timers so that
    // handshake isn't wedged behind the suite's fake clock.
    vi.useRealTimers()
    await ensureGatewayProfile('coder')
    vi.useFakeTimers()

    expect(isActivePrimary()).toBe(false)
    expect($activeGatewayProfile.get()).toBe('coder')
    expect($connection.get()?.profile).toBe('coder')
    expect($connection.get()?.baseUrl).toBe(coderConn.baseUrl)

    const callsBeforeDrop = desktop.getConnection.mock.calls.length
    const socketsBeforeDrop = FakeWebSocket.instances.length
    const primarySocket = FakeWebSocket.instances[0]

    act(() => primarySocket.drop())
    await flushAsync()
    await advanceBackoff()

    const reconnectCalls = desktop.getConnection.mock.calls.slice(callsBeforeDrop)
    expect(reconnectCalls.some(args => (args[0] ?? '').trim() === 'coder')).toBe(false)
    expect(reconnectCalls.some(args => args.length === 0 || args[0] == null || args[0] === '')).toBe(true)

    const primaryReconnectSockets = FakeWebSocket.instances
      .slice(socketsBeforeDrop)
      .filter(socket => socket.url === primaryConn.wsUrl)

    expect(primaryReconnectSockets.length).toBeGreaterThan(0)
    expect($connection.get()?.profile).toBe('coder')
    expect($connection.get()?.baseUrl).toBe(coderConn.baseUrl)
  })

  it('FIX #82679: a transient remote boot failure self-heals — the next attempt rebuilds the dropped connection', async () => {
    // The reported class: the app relaunches (or wakes) against a registered
    // SSH/HTTP remote whose transport dropped. startHermes() rejects with a
    // transient transport error ("Could not verify the existing SSH backend"),
    // main tags the boot progress `retryable`, and — before the fix — the app
    // parked on "Desktop boot failed" until the user re-entered the exact same
    // connection details. Now the renderer retries the boot with backoff and
    // the second attempt (fresh bootstrap, same details) succeeds.
    const desktop = fakeDesktop()
    desktop.getConnection = vi
      .fn()
      .mockRejectedValueOnce(new Error('Could not verify the existing SSH backend.'))
      .mockImplementation(async () => primaryConn)
    desktop.getBootProgress = vi.fn(async () => ({
      error: 'Could not verify the existing SSH backend.',
      fakeMode: false,
      message: 'Desktop boot failed: Could not verify the existing SSH backend.',
      phase: 'backend.error',
      progress: 24,
      retryable: true,
      running: false,
      timestamp: Date.now()
    }))
    ;(window as { hermesDesktop?: unknown }).hermesDesktop = desktop

    render(<Harness />)
    await flushAsync()

    // First attempt failed but the failure is retryable: no terminal error,
    // the overlay shows the retry status instead of the dead-end failure.
    expect($desktopBoot.get().error).toBeNull()
    expect($gatewayState.get()).not.toBe('open')

    // Walk past the first backoff delay (2s base, 15s cap, full jitter).
    await advanceBackoff()

    // Second boot attempt rebuilt the connection — no manual re-entry.
    expect(desktop.getConnection.mock.calls.length).toBeGreaterThan(1)
    expect($gatewayState.get()).toBe('open')
    expect($desktopBoot.get().error).toBeNull()
  })

  it('FIX #82679: boot retries are BOUNDED — a persistently dead remote ends in the recovery overlay, not a spinner', async () => {
    const desktop = fakeDesktop()
    desktop.getConnection = vi.fn(async () => {
      throw new Error('Could not verify the existing SSH backend.')
    })
    desktop.getBootProgress = vi.fn(async () => ({
      error: 'Could not verify the existing SSH backend.',
      fakeMode: false,
      message: 'Desktop boot failed: Could not verify the existing SSH backend.',
      phase: 'backend.error',
      progress: 24,
      retryable: true,
      running: false,
      timestamp: Date.now()
    }))
    ;(window as { hermesDesktop?: unknown }).hermesDesktop = desktop

    render(<Harness />)
    await flushAsync()

    // Exhaust the bounded retry budget (5 attempts, ≤15s jittered delay each).
    for (let i = 0; i < 7; i += 1) {
      await advanceBackoff()
    }

    // 1 initial + 5 bounded retries; the loop then STOPS retrying and the
    // terminal boot error surfaces the real recovery affordance.
    expect(desktop.getConnection).toHaveBeenCalledTimes(6)
    expect($desktopBoot.get().error).toBeTruthy()

    // No further attempts after the budget is spent — bounded, not infinite.
    await advanceBackoff()
    expect(desktop.getConnection).toHaveBeenCalledTimes(6)
  })

  it('FIX #82679: a NON-retryable boot failure (local / confirmed reauth) fails immediately without auto-retry', async () => {
    const desktop = fakeDesktop()
    desktop.getConnection = vi.fn(async () => {
      throw new Error('401: gateway session expired')
    })
    desktop.getBootProgress = vi.fn(async () => ({
      error: '401: gateway session expired',
      fakeMode: false,
      message: 'Desktop boot failed: 401: gateway session expired',
      phase: 'backend.error',
      progress: 24,
      retryable: false,
      running: false,
      timestamp: Date.now()
    }))
    ;(window as { hermesDesktop?: unknown }).hermesDesktop = desktop

    render(<Harness />)
    await flushAsync()

    expect($desktopBoot.get().error).toBeTruthy()
    expect(desktop.getConnection).toHaveBeenCalledTimes(1)

    // Still no retry later: a missing capability is not a transient failure.
    await advanceBackoff()
    expect(desktop.getConnection).toHaveBeenCalledTimes(1)
  })

  it('wake probe: an open-looking but unresponsive socket is force-closed and reconnected', async () => {
    render(<Harness />)
    await flushAsync()
    expect($gatewayState.get()).toBe('open')
    const socketCountBefore = FakeWebSocket.instances.length

    // Half-open socket: connectionState reads 'open' (no close event) but the
    // backend never answers — the sleep/wake TCP black hole.
    FakeWebSocket.pingMode = 'silent'

    // A wake signal (power resume / network online / window visible) nudges
    // reconnectNow. With the socket still reporting open it must PROBE rather
    // than skip; the swallowed ping times out and forces the socket down.
    act(() => window.dispatchEvent(new Event('online')))
    await act(async () => {
      await vi.advanceTimersByTimeAsync(5_100)
    })
    // The probe timeout (5s) force-closed the socket → 'closed' → the backoff
    // timer schedules a reconnect; let it fire and re-dial.
    await advanceBackoff()

    // A fresh socket was dialed.
    expect(FakeWebSocket.instances.length).toBeGreaterThan(socketCountBefore)
    expect($gatewayState.get()).toBe('open')
  })

  it('wake probe: a healthy socket answers the ping and stays untouched', async () => {
    render(<Harness />)
    await flushAsync()
    expect($gatewayState.get()).toBe('open')
    const socketCountBefore = FakeWebSocket.instances.length

    // Default FakeWebSocket behavior: answer pings with a pong frame.
    act(() => window.dispatchEvent(new Event('online')))
    await act(async () => {
      await vi.advanceTimersByTimeAsync(5_100)
    })

    // Probe succeeded → no forced close, no reconnect, connection untouched.
    expect(FakeWebSocket.instances.length).toBe(socketCountBefore)
    expect($gatewayState.get()).toBe('open')
  })

  it('wake probe: a pre-ping backend (-32601) is healthy, not reconnected', async () => {
    render(<Harness />)
    await flushAsync()
    expect($gatewayState.get()).toBe('open')
    const socketCountBefore = FakeWebSocket.instances.length

    // Version skew: this gateway predates the ping method. The error response
    // proves the socket is alive; forcing a reconnect would loop forever.
    FakeWebSocket.pingMode = 'method-not-found'

    act(() => window.dispatchEvent(new Event('online')))
    await act(async () => {
      await vi.advanceTimersByTimeAsync(5_100)
    })

    expect(FakeWebSocket.instances.length).toBe(socketCountBefore)
    expect($gatewayState.get()).toBe('open')
  })
})
