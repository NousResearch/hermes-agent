import { act, cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

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
let connectionApplied: null | (() => void) = null

// Minimal WebSocket stand-in implementing only what json-rpc-gateway.connect()
// touches: readyState, add/removeEventListener('open'|'error'|'close'), close().
class FakeWebSocket {
  static OPEN = 1
  static CLOSED = 3
  // Flipped by the test: 'open' = next socket connects; 'fail' = next socket
  // errors (a dead remote); 'closeCode' = the gateway accepts the WS upgrade
  // and immediately closes with FakeWebSocket.closeCode instead of firing
  // 'error' (the hermes_cli/web_server.py auth-rejection shape).
  static mode: 'closeCode' | 'fail' | 'open' = 'open'
  static closeCode = 4401
  static instances: FakeWebSocket[] = []

  readyState = 0
  private listeners: Record<string, Set<Listener>> = {}

  constructor(public url: string) {
    FakeWebSocket.instances.push(this)
    const mode = FakeWebSocket.mode
    // Resolve on the next microtask/macrotask so connect()'s promise wiring is
    // in place before open/error/close fires (matches real async handshake).
    setTimeout(() => {
      if (mode === 'open') {
        this.readyState = FakeWebSocket.OPEN
        this.emit('open', {})
      } else if (mode === 'closeCode') {
        this.readyState = FakeWebSocket.CLOSED
        this.emit('close', { code: FakeWebSocket.closeCode })
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
    onBootProgress: vi.fn(() => () => undefined),
    onBackendExit: vi.fn(() => () => undefined),
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
  FakeWebSocket.closeCode = 4401
  FakeWebSocket.instances = []
  connectionApplied = null
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

  it('BOOT FIX: a transient connect failure during boot retries instead of failing immediately', async () => {
    // The remote-gateway version of this bug: the route (e.g. over Tailscale)
    // just isn't up yet at launch. Stock boot() had zero retries here — one
    // failed connect and the app went straight to the fatal overlay.
    FakeWebSocket.mode = 'fail'
    render(<Harness />)
    await flushAsync()

    // First attempt failed, but that alone must not be fatal.
    expect($gatewayState.get()).not.toBe('open')
    expect($desktopBoot.get().error).toBeNull()
    expect(FakeWebSocket.instances).toHaveLength(1)

    // The route comes up before the first scheduled retry fires.
    FakeWebSocket.mode = 'open'
    await advanceBackoff()

    expect($gatewayState.get()).toBe('open')
    expect($desktopBoot.get().error).toBeNull()
    expect(FakeWebSocket.instances.length).toBeGreaterThan(1)
  })

  it('BOOT FIX: an auth rejection during boot fails fast without retrying', async () => {
    // Distinct from the transport-failure case above: a stale/rejected OAuth
    // ticket can never succeed by retrying, so this must still fail on the
    // first attempt exactly like stock boot() did.
    const desktop = fakeDesktop()
    desktop.getConnection = vi.fn(async () => ({
      authMode: 'oauth' as const,
      baseUrl: 'https://vps.example.com',
      profile: 'default',
      token: '',
      wsUrl: 'wss://vps.example.com/api/ws?ticket=stale'
    })) as unknown as typeof desktop.getConnection
    desktop.getGatewayWsUrl = vi.fn(async () => ({
      error: 'ticket expired',
      needsOauthLogin: true,
      ok: false as const
    })) as unknown as typeof desktop.getGatewayWsUrl
    ;(window as { hermesDesktop?: unknown }).hermesDesktop = desktop

    render(<Harness />)
    await flushAsync()

    // resolveGatewayWsUrl throws before gateway.connect() ever runs — no
    // socket was created.
    expect(FakeWebSocket.instances).toHaveLength(0)
    expect($desktopBoot.get().error).toBeTruthy()

    // No retry got scheduled: advancing well past the backoff window doesn't
    // draw another getConnection() call.
    await advanceBackoff()
    expect(desktop.getConnection).toHaveBeenCalledTimes(1)
  })

  it('BOOT FIX: a handshake refused with an auth close code (4401) fails fast without retrying', async () => {
    // The gateway can reject an unauthorized/expired session by accepting the
    // WS upgrade and immediately closing with an app-level auth code
    // (hermes_cli/web_server.py) instead of ever firing 'error'. That must
    // fail fast like the stale-ticket case above, not sit through the
    // connect timeout and then retry for ~45s behind the same overlay.
    FakeWebSocket.mode = 'closeCode'
    FakeWebSocket.closeCode = 4401
    render(<Harness />)
    await flushAsync()

    expect(FakeWebSocket.instances).toHaveLength(1)
    expect($desktopBoot.get().error).toBeTruthy()
    // Not just "some error" — it must be the sign-in-recovery text
    // isRemoteReauthError() string-matches on (boot-failure-reauth.ts), or the
    // overlay drops into the generic local-only Retry/Repair buttons instead
    // of the actual fix (sign in again).
    expect($desktopBoot.get().error).toMatch(/remote gateway session has expired/i)

    // No retry got scheduled: advancing well past the backoff window draws no
    // second socket.
    await advanceBackoff()
    expect(FakeWebSocket.instances).toHaveLength(1)
  })

  it('BOOT FIX: a handshake refused with a policy close code (4403) fails fast but is not sign-in-shaped', async () => {
    // 4403 means the gateway rejected the connection on host/origin/policy
    // grounds (embedded chat disabled, wrong Host header, etc. —
    // web_server.py:18632,18640), not an expired session. It must still fail
    // fast (retrying a server-issued refusal is pointless) but must NOT carry
    // needsOauthLogin or the sign-in message — re-authenticating can't fix it,
    // so routing to the reauth overlay would be actively misleading.
    FakeWebSocket.mode = 'closeCode'
    FakeWebSocket.closeCode = 4403
    render(<Harness />)
    await flushAsync()

    expect(FakeWebSocket.instances).toHaveLength(1)
    expect($desktopBoot.get().error).toBeTruthy()
    expect($desktopBoot.get().error).not.toMatch(/remote gateway session has expired/i)
    expect($desktopBoot.get().error).toBe('Could not connect to Hermes gateway')

    // No retry got scheduled.
    await advanceBackoff()
    expect(FakeWebSocket.instances).toHaveLength(1)
  })

  it('BOOT FIX: repeated transient failures escalate to a recoverable error, and retries continue afterward', async () => {
    FakeWebSocket.mode = 'fail'
    render(<Harness />)
    await flushAsync()
    expect($desktopBoot.get().error).toBeNull()

    // Walk the backoff past the escalation threshold (mirrors the post-boot
    // reconnect loop's own >=6-attempt walk above).
    for (let i = 0; i < 8; i += 1) {
      await advanceBackoff()
    }

    expect($desktopBoot.get().error).toBeTruthy()
    const attemptsAtEscalation = FakeWebSocket.instances.length
    expect(attemptsAtEscalation).toBeGreaterThan(1)

    // Retrying must not stop once the overlay is up — a later attempt (the
    // "7th try") still runs and can recover.
    FakeWebSocket.mode = 'open'
    await advanceBackoff()

    expect($gatewayState.get()).toBe('open')
    expect($desktopBoot.get().error).toBeNull()
    expect(FakeWebSocket.instances.length).toBeGreaterThan(attemptsAtEscalation)
  })

  it('BOOT FIX: a pending boot retry does not re-dial after a soft gateway switch', async () => {
    // The race: boot() schedules a retry, then before it fires the user
    // applies a different gateway (softSwitch, driven by connectionApplied).
    // A stale retry that re-enters boot() here would call gateway.connect()
    // while the switch's own dial already flipped the socket to 'connecting'
    // (or past it, to 'open') — connect() short-circuits on that state without
    // awaiting a real socket, so boot() would treat the connection as open
    // before it actually is and re-run completeDesktopBoot() behind the
    // switch's back.
    const desktop = fakeDesktop()
    ;(window as { hermesDesktop?: unknown }).hermesDesktop = desktop

    FakeWebSocket.mode = 'fail'
    render(<Harness />)
    await flushAsync()

    // Initial boot failed once; a retry is pending (not yet fired).
    expect(FakeWebSocket.instances).toHaveLength(1)
    expect($desktopBoot.get().error).toBeNull()
    const dialsBeforeSwitch = desktop.getConnection.mock.calls.length

    // The remote comes up before the switch dials, so the switch itself
    // succeeds cleanly — isolating the assertion to "did the stale retry
    // re-fire", not "did the switch also fail".
    FakeWebSocket.mode = 'open'
    act(() => connectionApplied?.())
    await flushAsync()

    expect($gatewayState.get()).toBe('open')
    // softSwitch's own dial: exactly one new socket, one new getConnection call.
    expect(FakeWebSocket.instances).toHaveLength(2)
    expect(desktop.getConnection.mock.calls.length).toBe(dialsBeforeSwitch + 1)

    // Advance well past where the stale retry would have fired (1s) — nothing
    // new dials.
    await advanceBackoff()

    expect(FakeWebSocket.instances).toHaveLength(2)
    expect(desktop.getConnection.mock.calls.length).toBe(dialsBeforeSwitch + 1)
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
