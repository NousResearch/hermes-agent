import { act, cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { BackendExit, DesktopBootProgress } from '@/global'
import { $desktopBoot } from '@/store/boot'
import { $gatewayState } from '@/store/session'

import { takeGatewaySurvivor } from './gateway-hmr-survivor'
import { useGatewayBoot } from './use-gateway-boot'

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
let backendExit: null | ((payload: BackendExit) => void) = null
let connectionApplied: null | (() => void) = null
let bootProgress: null | ((payload: DesktopBootProgress) => void) = null

// Minimal WebSocket stand-in implementing only what json-rpc-gateway.connect()
// touches: readyState, add/removeEventListener('open'|'error'|'close'), close().
class FakeWebSocket {
  static OPEN = 1
  static CLOSED = 3
  // Flipped by the test: 'open' = next socket connects; 'fail' = next socket
  // errors (a dead remote). Mirrors a VPS going away after the first connect.
  static mode: 'open' | 'fail' = 'open'
  static instances: FakeWebSocket[] = []

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

  private emit(type: string, ev: unknown) {
    for (const fn of this.listeners[type] ?? []) {
      fn(ev)
    }
  }
}

function fakeDesktop() {
  const conn = {
    authMode: 'token' as const,
    baseUrl: 'https://vps.example.com',
    profile: 'default',
    token: 't',
    wsUrl: 'wss://vps.example.com/api/ws?token=t'
  }

  return {
    getConnection: vi.fn(async () => conn),
    getGatewayWsUrl: vi.fn(async () => conn.wsUrl),
    getBootProgress: vi.fn(async () => ({
      error: null,
      fakeMode: false,
      message: '',
      phase: 'init',
      progress: 0,
      running: true,
      timestamp: Date.now()
    })),
    onBootProgress: vi.fn(callback => {
      bootProgress = callback

      return () => {
        bootProgress = null
      }
    }),
    onBackendExit: vi.fn(callback => {
      backendExit = callback

      return () => {
        backendExit = null
      }
    }),
    onConnectionApplied: vi.fn(callback => {
      connectionApplied = callback

      return () => {
        connectionApplied = null
      }
    }),
    onPowerResume: vi.fn(() => () => undefined),
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

  vi.useFakeTimers()
  FakeWebSocket.mode = 'open'
  FakeWebSocket.instances = []
  backendExit = null
  connectionApplied = null
  bootProgress = null
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

  vi.useRealTimers()
  ;(globalThis as { WebSocket: unknown }).WebSocket = originalWebSocket
  delete (window as { hermesDesktop?: unknown }).hermesDesktop
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

// One replay of the two steps startHermes() emits whenever it re-enters its
// remote branch (electron/main.ts: advanceBootProgress 'backend.resolve' then
// 'backend.remote', both running:true / error:null). The main process re-enters
// it for any caller that asks for a connection, boot or no boot.
function emitRemoteBootRetry() {
  act(() => {
    bootProgress?.({
      error: null,
      fakeMode: false,
      message: 'Resolving Hermes backend',
      phase: 'backend.resolve',
      progress: 8,
      running: true,
      timestamp: Date.now()
    })
    bootProgress?.({
      error: null,
      fakeMode: false,
      message: 'Connecting to remote Hermes backend at https://vps.example.com',
      phase: 'backend.remote',
      progress: 24,
      running: true,
      timestamp: Date.now()
    })
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

  it('FIX: the failed boot keeps its recovery surface while the main process retries behind it', async () => {
    // Where the test above stops, one event too early. boot() runs once, but
    // the main process keeps startHermes() available, and every later caller
    // that wants a connection re-enters it and replays the cold-boot progress
    // steps. They land on a renderer whose boot is already over: the first
    // running:true hides the recovery overlay, the second wipes boot.error, and
    // the fullscreen CONNECTING screen covers the app for the rest of that
    // attempt. Nothing concludes this boot a second time, so only the main
    // process's own failure payload could bring the recovery surface back.
    const desktop = fakeDesktop()
    desktop.getConnection = vi.fn(async () => {
      throw new Error('Hermes backend did not become ready: connect ECONNREFUSED 127.0.0.1:9119')
    })
    ;(window as { hermesDesktop?: unknown }).hermesDesktop = desktop

    render(<Harness />)
    await flushAsync()

    expect($desktopBoot.get().error).toBeTruthy()

    emitRemoteBootRetry()
    emitRemoteBootRetry()

    // The pair BootFailureOverlay renders on (error set, nothing running), and
    // the same boot.error that keeps GatewayConnectingOverlay off the screen.
    expect($desktopBoot.get().error).toBeTruthy()
    expect($desktopBoot.get().running).toBe(false)
  })

  it('FIX: a stale boot snapshot cannot hide the recovery surface after boot fails', async () => {
    const desktop = fakeDesktop()
    type BootSnapshot = Awaited<ReturnType<typeof desktop.getBootProgress>>
    let resolveSnapshot: (snapshot: BootSnapshot) => void = () => undefined

    desktop.getBootProgress = vi.fn(
      () =>
        new Promise<BootSnapshot>(resolve => {
          resolveSnapshot = resolve
        })
    )
    desktop.getConnection = vi.fn(async () => {
      throw new Error('Hermes backend did not become ready: connect ECONNREFUSED 127.0.0.1:9119')
    })
    ;(window as { hermesDesktop?: unknown }).hermesDesktop = desktop

    render(<Harness />)
    await flushAsync()

    expect($desktopBoot.get().error).toBeTruthy()

    await act(async () => {
      resolveSnapshot({
        error: null,
        fakeMode: false,
        message: 'Resolving Hermes backend',
        phase: 'backend.resolve',
        progress: 8,
        running: true,
        timestamp: Date.now()
      })
      await vi.advanceTimersByTimeAsync(0)
    })

    expect($desktopBoot.get().error).toBeTruthy()
    expect($desktopBoot.get().running).toBe(false)
  })

  it('FIX: the same holds when the boot fails at the gateway socket instead of getConnection', async () => {
    // The other way a cold boot ends badly, and the one a stale token produces:
    // the backend answers and reports ready, so boot() gets past getConnection,
    // and the gateway socket is what refuses. It concludes in the same catch, so
    // the recovery surface has to survive the same replayed progress.
    FakeWebSocket.mode = 'fail'

    render(<Harness />)
    await flushAsync()

    expect($desktopBoot.get().error).toBeTruthy()

    emitRemoteBootRetry()

    expect($desktopBoot.get().error).toBeTruthy()
    expect($desktopBoot.get().running).toBe(false)
  })

  it('FIX: the same holds when the backend exits during startup and onBackendExit raises the failure', async () => {
    // The third way a boot ends in failure, and the only one that does not run
    // inside boot()'s own catch: a local backend that dies while boot() is
    // still awaiting a connection. onBackendExit concludes the boot on its
    // behalf, and the main process restarting that backend is exactly what
    // replays the progress steps, so this conclusion has to latch as well.
    const desktop = fakeDesktop()
    // boot() is still awaiting getConnection when the process goes away.
    desktop.getConnection = vi.fn(() => new Promise(() => undefined))
    ;(window as { hermesDesktop?: unknown }).hermesDesktop = desktop

    render(<Harness />)
    await flushAsync()

    expect($desktopBoot.get().running).toBe(true)

    act(() => backendExit?.({ code: 1, signal: null }))

    expect($desktopBoot.get().error).toBeTruthy()

    emitRemoteBootRetry()

    expect($desktopBoot.get().error).toBeTruthy()
    expect($desktopBoot.get().running).toBe(false)
  })

  it('FIX: the failed boot is not a dead end — applying a gateway config still boots the app', async () => {
    // The latch must not outlive the failure it describes. "Use local gateway"
    // and the embedded gateway settings both apply a config, which drives
    // softSwitch(): a fresh boot lifecycle that has to clear the failure and
    // open the app.
    let connectionFails = true
    const desktop = fakeDesktop()
    const healthyConnection = desktop.getConnection

    desktop.getConnection = vi.fn(async () => {
      if (connectionFails) {
        throw new Error('Hermes backend did not become ready: connect ECONNREFUSED 127.0.0.1:9119')
      }

      return healthyConnection()
    })
    ;(window as { hermesDesktop?: unknown }).hermesDesktop = desktop

    render(<Harness />)
    await flushAsync()

    emitRemoteBootRetry()
    expect($desktopBoot.get().error).toBeTruthy()

    connectionFails = false
    act(() => connectionApplied?.())
    await flushAsync()

    expect($gatewayState.get()).toBe('open')
    expect($desktopBoot.get().error).toBeNull()
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

    // Burn a couple backoff cycles BEFORE the escalation threshold (<6 attempts,
    // ~the first ~15s). This is the window where stock and fixed behave the
    // same: socket down, hook retrying, gatewayState non-open, boot.error still
    // null → CONNECTING covers the screen with no recovery surface. (Past ~45s
    // the fix raises boot.error; that's asserted in the next test.)
    await advanceBackoff()

    expect($gatewayState.get()).not.toBe('open')
    expect($desktopBoot.get().error).toBeNull()
    // It is actively retrying, not idle — more sockets were minted.
    expect(FakeWebSocket.instances.length).toBeGreaterThan(1)
  })

  it('FIX: after the prolonged drop the hook raises a recoverable boot error (the escape hatch)', async () => {
    render(<Harness />)
    await flushAsync()
    expect($desktopBoot.get().error).toBeNull()

    FakeWebSocket.mode = 'fail'
    act(() => FakeWebSocket.instances[0].drop())
    await flushAsync()

    // Walk the backoff past the >=6 attempt threshold (~45s of failures).
    for (let i = 0; i < 8; i += 1) {
      await advanceBackoff()
    }

    // The hook surfaced the recoverable error → BootFailureOverlay (Use local
    // gateway / Sign in / Retry) becomes reachable instead of CONNECTING.
    expect($desktopBoot.get().error).toBeTruthy()
  })

  it('FIX: a successful reconnect clears the recoverable error', async () => {
    render(<Harness />)
    await flushAsync()

    FakeWebSocket.mode = 'fail'
    act(() => FakeWebSocket.instances[0].drop())
    await flushAsync()

    for (let i = 0; i < 8; i += 1) {
      await advanceBackoff()
    }

    expect($desktopBoot.get().error).toBeTruthy()

    // The remote comes back: next reconnect attempt opens.
    FakeWebSocket.mode = 'open'
    await advanceBackoff()

    expect($gatewayState.get()).toBe('open')
    expect($desktopBoot.get().error).toBeNull()
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
})
