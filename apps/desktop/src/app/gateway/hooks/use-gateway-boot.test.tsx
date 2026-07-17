import { act, cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesConnection } from '@/global'
import { $desktopBoot } from '@/store/boot'
import { $activeGatewayProfile } from '@/store/profile'
import {
  $connection,
  $gatewayState,
  $workingSessionIds,
  $workingSessionProfiles,
  setSessionWorking
} from '@/store/session'
import { $sessionActivityKeys, sessionActivityKey } from '@/store/session-activity'
import { $subagentsBySession, upsertSubagent } from '@/store/subagents'

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

// Minimal WebSocket stand-in implementing only what json-rpc-gateway.connect()
// touches: readyState, add/removeEventListener('open'|'error'|'close'), close().
class FakeWebSocket {
  static OPEN = 1
  static CLOSED = 3
  // Flipped by the test: 'open' = next socket connects; 'fail' = next socket
  // errors (a dead remote). Mirrors a VPS going away after the first connect.
  static mode: 'open' | 'fail' = 'open'
  static instances: FakeWebSocket[] = []
  static delegationActive: unknown[] = []
  static activeSessions: unknown[] = []
  static activeSessionsByProfile: Record<string, unknown[]> = {}
  static requestedMethods: string[] = []

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

  send(raw: string) {
    const frame = JSON.parse(raw) as { id?: number | string; method?: string }

    if (frame.method) {
      FakeWebSocket.requestedMethods.push(frame.method)
    }

    if (frame.method === 'delegation.status') {
      setTimeout(() => {
        this.emit('message', {
          data: JSON.stringify({ id: frame.id, result: { active: FakeWebSocket.delegationActive } })
        })
      }, 0)
    } else if (frame.method === 'session.active_list') {
      const profile = new URL(this.url).searchParams.get('profile')
      const sessions = (profile && FakeWebSocket.activeSessionsByProfile[profile]) ?? FakeWebSocket.activeSessions

      queueMicrotask(() => {
        this.emit('message', {
          data: JSON.stringify({ id: frame.id, result: { sessions } })
        })
      })
    }
  }

  pushEvent(event: { payload?: unknown; session_id?: string; type: string }) {
    this.emit('message', { data: JSON.stringify({ method: 'event', params: event }) })
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

function connectionFor(profile = 'default'): HermesConnection {
  return {
    authMode: 'token' as const,
    baseUrl: `https://${profile}.example.com`,
    isFullscreen: false,
    logs: [],
    nativeOverlayWidth: 0,
    profile,
    token: 't',
    windowButtonPosition: null,
    wsUrl: `wss://${profile}.example.com/api/ws?profile=${profile}&token=t`
  }
}

function fakeDesktop() {
  const conn = connectionFor()

  return {
    getConnection: vi.fn(async (_profile?: null | string) => conn),
    getGatewayWsUrl: vi.fn(async (profile?: null | string) => connectionFor(profile ?? 'default').wsUrl),
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
    onConnectionApplied: vi.fn(() => () => undefined),
    onPowerResume: vi.fn(() => () => undefined),
    onWindowStateChanged: vi.fn(() => () => undefined),
    touchBackend: vi.fn(async () => undefined),
    profile: { get: vi.fn(async () => ({ profile: 'default' })) }
  }
}

const capturedEvents: Array<{ profile?: string; session_id?: string; type: string }> = []

function Harness() {
  useGatewayBoot({
    handleGatewayEvent: event => capturedEvents.push(event),
    onConnectionReady: () => undefined,
    onGatewayReady: () => undefined,
    refreshHermesConfig: async () => undefined,
    refreshSessions: async () => undefined
  })

  return null
}

const originalWebSocket = globalThis.WebSocket

beforeEach(() => {
  vi.useFakeTimers()
  FakeWebSocket.mode = 'open'
  FakeWebSocket.instances = []
  FakeWebSocket.delegationActive = []
  FakeWebSocket.activeSessions = []
  FakeWebSocket.activeSessionsByProfile = {}
  FakeWebSocket.requestedMethods = []
  capturedEvents.length = 0
  ;(globalThis as { WebSocket: unknown }).WebSocket = FakeWebSocket
  ;(window as { hermesDesktop?: unknown }).hermesDesktop = fakeDesktop()
  $gatewayState.set('idle')
  $activeGatewayProfile.set('default')
  $connection.set(null)
  $workingSessionIds.set([])
  $workingSessionProfiles.set({})
  $subagentsBySession.set({})
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
  it('tags primary events with the named profile adopted before the socket opens', async () => {
    const desktop = fakeDesktop()
    desktop.profile.get = vi.fn(async () => ({ profile: 'work' }))
    ;(window as { hermesDesktop?: unknown }).hermesDesktop = desktop

    render(<Harness />)
    await flushAsync()

    act(() => {
      FakeWebSocket.instances[0].pushEvent({ session_id: 'runtime-work', type: 'subagent.start' })
    })

    expect(capturedEvents.at(-1)).toMatchObject({ profile: 'work', session_id: 'runtime-work' })
  })

  it('restores a live working indicator from the gateway snapshot after reconnect', async () => {
    const desktop = fakeDesktop()
    desktop.profile.get = vi.fn(async () => ({ profile: 'work' }))
    ;(window as { hermesDesktop?: unknown }).hermesDesktop = desktop
    FakeWebSocket.activeSessions = [
      { id: 'runtime-work', session_key: 'stored-work', status: 'working' },
      { id: 'runtime-starting', session_key: 'stored-starting', status: 'starting' },
      { id: 'runtime-waiting', session_key: 'stored-waiting', status: 'waiting' },
      { id: 'runtime-idle', session_key: 'stored-idle', status: 'idle' }
    ]

    render(<Harness />)
    await flushAsync()

    expect(FakeWebSocket.requestedMethods).toContain('session.active_list')
    expect($sessionActivityKeys.get()).toContain(sessionActivityKey('work', 'stored-work'))
    expect($sessionActivityKeys.get()).toContain(sessionActivityKey('work', 'stored-starting'))
    expect($sessionActivityKeys.get()).not.toContain(sessionActivityKey('work', 'stored-waiting'))
    expect($sessionActivityKeys.get()).not.toContain(sessionActivityKey('work', 'stored-idle'))
  })

  it('reconnects the primary through its owning profile after another profile becomes active', async () => {
    const desktop = fakeDesktop()

    desktop.profile.get = vi.fn(async () => ({ profile: 'alpha' }))
    desktop.getConnection = vi.fn(async (profile?: null | string) => connectionFor(profile ?? 'alpha'))
    FakeWebSocket.activeSessionsByProfile = { alpha: [], beta: [] }
    ;(window as { hermesDesktop?: unknown }).hermesDesktop = desktop

    render(<Harness />)
    await flushAsync()

    act(() => {
      $activeGatewayProfile.set('beta')
      $connection.set(connectionFor('beta'))
    })
    FakeWebSocket.activeSessionsByProfile.alpha = [
      { id: 'runtime-alpha', session_key: 'stored-alpha', status: 'working' }
    ]
    FakeWebSocket.activeSessionsByProfile.beta = [{ id: 'runtime-beta', session_key: 'stored-beta', status: 'working' }]

    act(() => FakeWebSocket.instances[0].drop())
    await advanceBackoff()

    expect(desktop.getConnection).toHaveBeenLastCalledWith('alpha')
    expect(FakeWebSocket.instances.at(-1)?.url).toContain('profile=alpha')
    expect($connection.get()?.profile).toBe('beta')
    expect($sessionActivityKeys.get()).toContain(sessionActivityKey('alpha', 'stored-alpha'))
    expect($sessionActivityKeys.get()).not.toContain(sessionActivityKey('alpha', 'stored-beta'))
  })

  it('keeps existing working activity when a reconnect snapshot contains a malformed row', async () => {
    setSessionWorking('existing', true, 'default')
    FakeWebSocket.activeSessions = [{}]

    render(<Harness />)
    await flushAsync()

    expect($sessionActivityKeys.get()).toContain(sessionActivityKey('default', 'existing'))
  })

  it('keeps existing activity when a reconnect snapshot contains a malformed row', async () => {
    upsertSubagent(
      'runtime-default',
      { detached: true, goal: 'live review', status: 'running', subagent_id: 'live' },
      true,
      'subagent.start',
      'owner-default',
      'default'
    )
    FakeWebSocket.delegationActive = [{}]

    render(<Harness />)
    await flushAsync()

    expect($subagentsBySession.get()['runtime-default']).toMatchObject([{ id: 'live', profile: 'default' }])
  })

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
})
