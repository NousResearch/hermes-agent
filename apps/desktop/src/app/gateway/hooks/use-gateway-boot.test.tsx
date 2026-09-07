import { act, cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { DesktopConnectionsRegistry } from '@/global'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $desktopBoot } from '@/store/boot'
import {
  $connectionsRegistry,
  _resetConnectionsForTests,
  selectConnection,
  setConnectionsRegistry
} from '@/store/connections'
import {
  activeGateway,
  closeSecondaryGateways,
  disposeSecondariesForConnection,
  ensureGatewayForAgent,
  isActivePrimary,
  requestGatewayForAgent
} from '@/store/gateway'
import { reconnectGateway } from '@/store/gateway-reconnect'
import {
  $gatewaySwitching,
  beginGatewaySwitch,
  endGatewaySwitch,
  recoverActiveSourceAfterFailedGatewaySwitch
} from '@/store/gateway-switch'
import { notifyError } from '@/store/notifications'
import { $activeGatewayProfile, $profiles, ensureGatewayProfile } from '@/store/profile'
import {
  $activeSessionId,
  $awaitingResponse,
  $busy,
  $connection,
  $currentCwd,
  $gatewayState,
  $selectedStoredSessionId,
  $sessionsLoading,
  getConfiguredDefaultProjectDir,
  setActiveSessionId,
  setSelectedStoredSessionId
} from '@/store/session'
import { $sessionTiles, $workingSessionIds, clearAllSessionStates, publishSessionState } from '@/store/session-states'

import { deferred } from '../../../test/deferred'

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
  refreshHermesConfig = async () => undefined,
  refreshSessions
}: {
  beforeConnectionSwitch?: () => void
  refreshHermesConfig?: (force?: boolean, shouldPublish?: () => boolean) => Promise<void>
  refreshSessions?: (shouldPublish?: () => boolean) => Promise<void>
} = {}) {
  useGatewayBoot({
    beforeConnectionSwitch,
    handleGatewayEvent: () => undefined,
    onConnectionReady: () => undefined,
    onGatewayReady: () => undefined,
    refreshHermesConfig,
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
  $busy.set(false)
  $awaitingResponse.set(false)
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
  $busy.set(false)
  $awaitingResponse.set(false)
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

  it('a stale failed Settings switch cannot publish failure or disarm the newer switch owner', async () => {
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

    expect(notifyError).not.toHaveBeenCalled()
    expect($desktopBoot.get().error).toBeNull()
    expect($gatewaySwitching.get()).toBe(true)
    expect($sessionsLoading.get()).toBe(true)

    await act(async () => {
      rejectB(failureB)
      await vi.advanceTimersByTimeAsync(0)
    })

    expect(notifyError).toHaveBeenCalledTimes(1)
    expect(notifyError).toHaveBeenCalledWith(failureB, expect.any(String))
    expect($desktopBoot.get().error).toBe(failureB.message)
    expect($gatewaySwitching.get()).toBe(false)
    expect($sessionsLoading.get()).toBe(false)
  })

  it('does not publish a late Settings failure after a newer switch wins', async () => {
    const desktop = fakeDesktop()

    let rejectStale: (error: Error) => void = () => undefined

    ;(window as { hermesDesktop?: unknown }).hermesDesktop = desktop
    render(<Harness />)
    await flushAsync()

    desktop.getConnection.mockImplementationOnce(
      () =>
        new Promise((_resolve, reject) => {
          rejectStale = reject
        })
    )

    act(() => connectionApplied?.())
    await vi.waitFor(() => expect(desktop.getConnection).toHaveBeenCalledTimes(2))
    act(() => connectionApplied?.())
    await flushAsync()
    await flushAsync()

    expect($gatewaySwitching.get()).toBe(false)
    expect($desktopBoot.get().error).toBeNull()

    await act(async () => {
      rejectStale(new Error('late stale failure'))
      await vi.advanceTimersByTimeAsync(0)
    })

    expect($desktopBoot.get().error).toBeNull()
    expect(notifyError).not.toHaveBeenCalled()
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

  it('a token-less setup failure cannot publish after a nested switch raised a newer barrier', async () => {
    const failure = new Error('outer setup failed')
    const beforeConnectionSwitch = vi.fn()
    let newerToken: null | ReturnType<typeof beginGatewaySwitch> = null

    beforeConnectionSwitch.mockImplementationOnce(() => {
      newerToken = beginGatewaySwitch()
      throw failure
    })

    render(<Harness beforeConnectionSwitch={beforeConnectionSwitch} />)
    await flushAsync()

    act(() => connectionApplied?.())
    await flushAsync()

    expect(newerToken).not.toBeNull()
    expect($gatewaySwitching.get()).toBe(true)
    expect($desktopBoot.get().error).toBeNull()
    expect(notifyError).not.toHaveBeenCalled()

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

    // Publishing the secondary must not relabel the primary socket. Returning
    // to its source should reuse that socket, not dial the secondary endpoint.
    const socketsAfterSwitch = FakeWebSocket.instances.length
    await expect(requestGatewayForAgent('primary-vps', 'default', 'ping')).resolves.toEqual({ pong: true })
    expect(FakeWebSocket.instances).toHaveLength(socketsAfterSwitch)
  })

  it('a Settings switch superseded while reading its descriptor cannot publish over a newer Sessions switch', async () => {
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

    const settingsConn = {
      ...primaryConn,
      connectionId: 'settings-a',
      profile: 'settings-profile',
      wsUrl: 'wss://settings-a.example.com/api/ws?token=a'
    }

    let releaseSettings: (connection: typeof settingsConn) => void = () => undefined

    desktop.api = vi.fn(async ({ path }: { path: string }) =>
      path === '/api/profiles/active' ? { active: 'coder', current: 'coder' } : { profiles: [] }
    )
    desktop.getConnectionFor = vi.fn(async ({ connectionId, profile }: { connectionId: string; profile: string }) => ({
      ...coderConn,
      connectionId,
      profile,
      registryScoped: true
    }))
    desktop.getGatewayWsUrlFor = vi.fn(async () => coderConn.wsUrl)
    desktop.connections = {
      list: vi.fn(async () => registryConnections),
      setLastUsed: vi.fn(async (id: string) => ({ ok: true, registry: { ...registryConnections, lastUsed: id } }))
    }
    ;(window as { hermesDesktop?: unknown }).hermesDesktop = desktop

    render(<Harness />)
    await flushAsync()
    expect($gatewayState.get()).toBe('open')

    setConnectionsRegistry(registryConnections)
    desktop.getConnection.mockImplementationOnce(
      () =>
        new Promise(resolve => {
          releaseSettings = resolve
        })
    )

    act(() => connectionApplied?.())
    await vi.waitFor(() => expect(desktop.getConnection).toHaveBeenCalledTimes(2))

    const sessionsSwitch = selectConnection('coder-remote')
    await flushAsync()
    await flushAsync()
    await flushAsync()
    await sessionsSwitch

    expect(isActivePrimary()).toBe(false)
    expect($activeGatewayProfile.get()).toBe('default')
    expect($connection.get()?.connectionId).toBe('coder-remote')

    const wsUrlReads = desktop.getGatewayWsUrl.mock.calls.length
    const profileReads = desktop.profile.get.mock.calls.length
    const profileRefreshes = vi.mocked(desktop.api as ReturnType<typeof vi.fn>).mock.calls.length
    const socketCount = FakeWebSocket.instances.length

    await act(async () => {
      releaseSettings(settingsConn)
      await vi.advanceTimersByTimeAsync(0)
    })

    // Switch-token ownership governs every later publication, not just loading
    // teardown: stale Settings work cannot publish/connect/refresh after B won.
    expect(isActivePrimary()).toBe(false)
    expect($activeGatewayProfile.get()).toBe('default')
    expect($connection.get()?.connectionId).toBe('coder-remote')
    expect(desktop.getGatewayWsUrl).toHaveBeenCalledTimes(wsUrlReads)
    expect(desktop.profile.get).toHaveBeenCalledTimes(profileReads)
    expect(desktop.api).toHaveBeenCalledTimes(profileRefreshes)
    expect(FakeWebSocket.instances).toHaveLength(socketCount)
  })

  it('passes switch ownership through a session refresh held across a newer switch', async () => {
    const staleRefresh = deferred<void>()
    const publications: string[] = []
    let switchRefresh = 0

    const refreshSessions = vi.fn(async (shouldPublish?: () => boolean) => {
      // Initial boot remains a compatible zero-argument caller.
      if (!shouldPublish) {
        return
      }

      switchRefresh += 1
      const label = switchRefresh === 1 ? 'settings-a' : 'settings-b'

      if (switchRefresh === 1) {
        await staleRefresh.promise
      }

      if (shouldPublish()) {
        publications.push(label)
      }
    })

    render(<Harness refreshSessions={refreshSessions} />)
    await flushAsync()

    act(() => connectionApplied?.())
    await vi.waitFor(() => expect(switchRefresh).toBe(1))

    act(() => connectionApplied?.())
    await vi.waitFor(() => expect(switchRefresh).toBe(2))
    await flushAsync()

    expect(publications).toEqual(['settings-b'])

    await act(async () => {
      staleRefresh.resolve()
      await vi.advanceTimersByTimeAsync(0)
    })

    expect(publications).toEqual(['settings-b'])
    expect($gatewaySwitching.get()).toBe(false)
  })

  it('forwards failed-switch recovery ownership through its registered lifecycle', async () => {
    const refreshSessions = vi.fn(async (_shouldPublish?: () => boolean) => undefined)

    render(<Harness refreshSessions={refreshSessions} />)
    await flushAsync()

    const failed = beginGatewaySwitch()

    recoverActiveSourceAfterFailedGatewaySwitch(failed)
    endGatewaySwitch(failed)
    await vi.waitFor(() => expect(refreshSessions).toHaveBeenCalledTimes(2))

    const shouldPublish = refreshSessions.mock.calls[1][0]

    expect(shouldPublish).toBeTypeOf('function')
    expect(shouldPublish?.()).toBe(true)

    const newer = beginGatewaySwitch()

    expect(shouldPublish?.()).toBe(false)
    endGatewaySwitch(newer)
  })

  it('a superseded Settings switch cannot publish delayed cwd or config work after the winner', async () => {
    const desktop = fakeDesktop() as ReturnType<typeof fakeDesktop> & Record<string, unknown>
    const staleSanitize = deferred<{ cwd: string }>()
    const staleConfig = deferred<void>()
    const configPublications: string[] = []
    let settingsRead = 0
    let sanitizeRead = 0
    let switchConfigRead = 0

    const settings = {
      getDefaultProjectDir: vi.fn(async () => {
        settingsRead += 1

        return {
          defaultLabel: settingsRead === 1 ? '/settings-a' : '/settings-b',
          dir: settingsRead === 1 ? '/settings-a' : '/settings-b',
          resolvedCwd: settingsRead === 1 ? '/settings-a' : '/settings-b'
        }
      })
    }

    const sanitizeWorkspaceCwd = vi.fn((cwd: string) => {
      sanitizeRead += 1

      return sanitizeRead === 1 ? staleSanitize.promise : Promise.resolve({ cwd })
    })

    ;(window as { hermesDesktop?: unknown }).hermesDesktop = desktop

    const refreshHermesConfig = async (_force = false, shouldPublish?: () => boolean) => {
      if (!shouldPublish) {
        return
      }

      switchConfigRead += 1
      const label = switchConfigRead === 1 ? 'settings-a' : 'settings-b'

      if (switchConfigRead === 1) {
        await staleConfig.promise
      }

      if (shouldPublish()) {
        configPublications.push(label)
      }
    }

    render(<Harness refreshHermesConfig={refreshHermesConfig} />)
    await flushAsync()
    expect($gatewayState.get()).toBe('open')

    desktop.settings = settings
    desktop.sanitizeWorkspaceCwd = sanitizeWorkspaceCwd

    act(() => connectionApplied?.())
    await vi.waitFor(() => expect(sanitizeWorkspaceCwd).toHaveBeenCalledTimes(1))

    act(() => connectionApplied?.())
    await flushAsync()
    await flushAsync()

    expect($gatewaySwitching.get()).toBe(false)
    expect(getConfiguredDefaultProjectDir()).toBe('/settings-b')
    expect($currentCwd.get()).toBe('/settings-b')
    expect(configPublications).toEqual(['settings-b'])

    await act(async () => {
      staleSanitize.resolve({ cwd: '/settings-a/stale' })
      staleConfig.resolve()
      await vi.advanceTimersByTimeAsync(0)
    })

    expect(getConfiguredDefaultProjectDir()).toBe('/settings-b')
    expect($currentCwd.get()).toBe('/settings-b')
    expect(configPublications).toEqual(['settings-b'])
  })

  it('publishes the cold-boot primary registry identity for owned session RPCs', async () => {
    render(<Harness />)
    await flushAsync()

    expect($gatewayState.get()).toBe('open')
    expect(FakeWebSocket.instances).toHaveLength(1)

    await expect(requestGatewayForAgent('primary-vps', 'default', 'ping')).resolves.toEqual({ pong: true })
    expect(FakeWebSocket.instances).toHaveLength(1)
  })

  it('keeps registered source sockets alive during a legacy mode apply', async () => {
    const desktop = fakeDesktop() as ReturnType<typeof fakeDesktop> & {
      getConnectionFor: ReturnType<typeof vi.fn>
    }

    desktop.getConnectionFor = vi.fn(async ({ connectionId, profile }: { connectionId: string; profile: string }) => ({
      ...coderConn,
      connectionId,
      profile,
      wsUrl: `wss://${connectionId}.example.com/api/ws?token=r`
    }))
    ;(window as { hermesDesktop?: unknown }).hermesDesktop = desktop

    render(<Harness />)
    await flushAsync()
    expect($gatewayState.get()).toBe('open')

    let opening!: Promise<boolean>
    act(() => {
      opening = ensureGatewayForAgent('cloud', 'default')
    })
    await flushAsync()
    await opening

    const registeredGateway = activeGateway()
    expect(registeredGateway).not.toBeNull()
    expect(isActivePrimary()).toBe(false)

    act(() => connectionApplied?.())
    await flushAsync()
    await flushAsync()

    // Applying the legacy Local/Cloud mode must not close an independent v2
    // source. The foreground returns to the new primary, while the registered
    // socket remains reusable and cannot arm ws_orphan_reap on the old backend.
    expect(registeredGateway?.connectionState).toBe('open')
    expect(isActivePrimary()).toBe(true)
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

  it('onActiveConnectionInvalidated: a fallback getConnection() that hangs rejects on its own instead of latching $connection forever (#93454 sibling)', async () => {
    // Repro: the active connection is a registered secondary (e.g. a Bots-pane
    // source). It gets removed/invalidated (disposeSecondariesForConnection),
    // which falls back to redialing the primary profile via
    // desktop.getConnection(fallbackProfile) — the one getConnection() await
    // in this file the #93454 bound-every-IPC-round-trip sweep never reached.
    // A wedged main-process round-trip here must reject instead of leaving
    // $connection pointed at a promise that never settles.
    const desktop = fakeDesktop() as ReturnType<typeof fakeDesktop> & {
      getConnectionFor: ReturnType<typeof vi.fn>
    }

    desktop.getConnectionFor = vi.fn(async ({ connectionId, profile }: { connectionId: string; profile: string }) => ({
      ...coderConn,
      connectionId,
      profile
    }))
    ;(window as { hermesDesktop?: unknown }).hermesDesktop = desktop

    render(<Harness />)
    await flushAsync()
    expect($gatewayState.get()).toBe('open')
    expect($connection.get()).not.toBeNull()

    let opening!: Promise<boolean>

    act(() => {
      opening = ensureGatewayForAgent('cloud', 'default')
    })
    await flushAsync()
    await opening
    expect(isActivePrimary()).toBe(false)

    // The active secondary is about to be evicted; the fallback re-dial for
    // the primary profile hangs indefinitely.
    desktop.getConnection.mockImplementation(() => new Promise(() => undefined))

    act(() => {
      disposeSecondariesForConnection('cloud')
    })
    await flushAsync()

    expect(isActivePrimary()).toBe(true)
    expect(desktop.getConnection).toHaveBeenCalledWith('default')
    // Still stuck behind the hung fallback dial — the invalidation handler's
    // .then()/.catch() has not run yet.
    expect($connection.get()).not.toBeNull()

    // Advance past the internal reconnect-attempt timeout (20s) — the stalled
    // fallback getConnection() must reject so the handler's catch publishes
    // null, instead of latching $connection on a connection that will never
    // resolve.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(20_000)
    })

    expect($connection.get()).toBeNull()
  })

  it('a getConnection() that hangs on INITIAL boot rejects on its own after the reconnect-attempt timeout, not only when main eventually gives up (#93454)', async () => {
    // boot()'s getConnection() had no bound of its own — only main's own
    // eventual timeout (e.g. waitForHermes, ~45s) ever settled it. A wedge
    // that main never resolves (not even a rejection) must not hang
    // "Starting Hermes…" forever; the renderer needs to own its own bound
    // here too, same as attemptReconnect() and softSwitch().
    const desktop = fakeDesktop()
    desktop.getConnection = vi.fn(() => new Promise(() => undefined))
    ;(window as { hermesDesktop?: unknown }).hermesDesktop = desktop

    render(<Harness />)
    await flushAsync()

    expect($desktopBoot.get().error).toBeNull()

    // Advance past the shared backend-boot budget (45s) — the
    // stalled await must reject on its own so boot()'s catch runs instead of
    // waiting indefinitely on main.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(45_000)
    })

    expect($desktopBoot.get().error).toBeTruthy()
  })

  it('softSwitch(): a getConnection() that hangs on a connection-apply switch does not latch $gatewaySwitching forever (#93454)', async () => {
    // Repro: main applies a new connection (onConnectionApplied), softSwitch()
    // re-dials via getConnection(), and the IPC round-trip wedges. Without an
    // internal timeout, the try block never settles, so the `finally` that
    // clears $gatewaySwitching never runs — the switch UI stays frozen until
    // the app is restarted.
    const desktop = fakeDesktop()
    const originalGetConnection = desktop.getConnection
    let callCount = 0

    desktop.getConnection = vi.fn((profile?: null | string) => {
      callCount += 1

      // Initial boot succeeds; the switch triggered below hangs indefinitely.
      return callCount === 1 ? originalGetConnection(profile) : new Promise(() => undefined)
    })
    ;(window as { hermesDesktop?: unknown }).hermesDesktop = desktop

    render(<Harness />)
    await flushAsync()
    expect($gatewayState.get()).toBe('open')
    expect(connectionApplied).not.toBeNull()

    act(() => connectionApplied?.())
    await flushAsync()

    expect($gatewaySwitching.get()).toBe(true)

    // Advance past the shared backend-boot budget (45s) — the
    // stalled await must reject so the `finally` clears $gatewaySwitching
    // instead of latching the switch UI frozen forever.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(45_000)
    })

    expect($gatewaySwitching.get()).toBe(false)
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

  it('FIX: a successful reconnect retires the focused composer busy latch (#93059)', async () => {
    // Backend respawned mid-turn (auto-update, sleep/wake): the focused
    // composer's draft latches never get their terminal busy:false, and Send
    // silently no-ops behind the busy guard until restart (#93059).
    render(<Harness />)
    await flushAsync()
    expect($gatewayState.get()).toBe('open')

    // A turn was mid-flight when the backend went away.
    act(() => {
      $busy.set(true)
      $awaitingResponse.set(true)
    })

    act(() => FakeWebSocket.instances[0].drop())
    await flushAsync()

    // The respawned backend answers the next dial.
    await advanceBackoff()

    expect($gatewayState.get()).toBe('open')
    expect($busy.get()).toBe(false)
    expect($awaitingResponse.get()).toBe(false)
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

  // #95327: every focus/visibility/power-resume nudge probes an OPEN socket
  // and force-closes it when the liveness ping times out. A backend that is
  // merely BUSY (a long silent tool call holding the loop) fails that probe
  // without being dead — closing the socket mid-turn is exactly what feeds the
  // gateway's ws_orphan_reap interrupt ("Operation interrupted." placeholder).
  // While any session still reports working, one inconclusive timeout must
  // defer the teardown (bounded re-probe) instead of killing the transport.
  it('wake probe: a timeout while a turn is IN FLIGHT defers the force-close', async () => {
    render(<Harness />)
    await flushAsync()
    expect($gatewayState.get()).toBe('open')
    const socketCountBefore = FakeWebSocket.instances.length

    // A turn is running on this very socket; backend silence is expected until
    // the tool call returns.
    act(() => {
      publishSessionState('rt-live-turn', {
        ...createClientSessionState(null),
        storedSessionId: 's-live-turn',
        busy: true
      })
    })
    expect($workingSessionIds.get()).toContain('s-live-turn')

    // Busy-but-alive: the ping is swallowed (loop starved), not refused.
    FakeWebSocket.pingMode = 'silent'

    act(() => window.dispatchEvent(new Event('online')))
    await act(async () => {
      await vi.advanceTimersByTimeAsync(5_100)
    })

    // First inconclusive probe with work in flight: the socket must survive —
    // not merely "some socket is open again after a teardown + redial", but
    // THIS incarnation, whose transcript stream the running turn rides on.
    expect($gatewayState.get()).toBe('open')
    const survivingSocket = FakeWebSocket.instances[socketCountBefore - 1]

    expect(survivingSocket.readyState).toBe(FakeWebSocket.OPEN)

    clearAllSessionStates()

    // Recovery must not wedge once the working flag is gone: persistent
    // silence still exhausts the streak and rebuilds the transport.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(45_000)
    })

    FakeWebSocket.pingMode = 'pong'
    await act(async () => {
      await vi.advanceTimersByTimeAsync(20_000)
    })

    expect($gatewayState.get()).toBe('open')
    expect(FakeWebSocket.instances.length).toBeGreaterThanOrEqual(socketCountBefore)
  })

  it('wake probe: repeated timeouts while busy still rebuild the socket (no deadlock)', async () => {
    render(<Harness />)
    await flushAsync()
    expect($gatewayState.get()).toBe('open')
    const socketCountBefore = FakeWebSocket.instances.length

    act(() => {
      publishSessionState('rt-live-turn-2', {
        ...createClientSessionState(null),
        storedSessionId: 's-live-turn-2',
        busy: true
      })
    })

    // Genuinely dead under the working flag: EVERY probe keeps timing out.
    FakeWebSocket.pingMode = 'silent'

    for (let nudge = 0; nudge < 3; nudge += 1) {
      act(() => window.dispatchEvent(new Event('online')))
      await act(async () => {
        await vi.advanceTimersByTimeAsync(10_000)
      })
    }

    clearAllSessionStates()

    // The streak guard only DELAYS the teardown; a persistently unresponsive
    // socket is still rebuilt rather than trusted forever.
    expect(FakeWebSocket.instances.length).toBeGreaterThan(socketCountBefore)

    FakeWebSocket.pingMode = 'pong'
    await act(async () => {
      await vi.advanceTimersByTimeAsync(20_000)
    })
    expect($gatewayState.get()).toBe('open')
  })
})
