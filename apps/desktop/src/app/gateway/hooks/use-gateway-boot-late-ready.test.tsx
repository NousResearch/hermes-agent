import { act, cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $desktopBoot } from '@/store/boot'
import { _resetConnectionsForTests } from '@/store/connections'
import { closeSecondaryGateways } from '@/store/gateway'
import { endGatewaySwitch } from '@/store/gateway-switch'
import { notifyError } from '@/store/notifications'
import { $activeGatewayProfile } from '@/store/profile'
import {
  $awaitingResponse,
  $busy,
  $connection,
  $currentCwd,
  $gatewayState,
  setActiveSessionId,
  setSelectedStoredSessionId
} from '@/store/session'

import { takeGatewaySurvivor } from './gateway-hmr-survivor'
import { useGatewayBoot } from './use-gateway-boot'

vi.mock(import('@/store/notifications'), async importOriginal => ({
  ...(await importOriginal()),
  notifyError: vi.fn()
}))

// Regression guard for #96743.
//
// Real-world sequence (SSH remote mode, after the one-click update flow):
//
//   1. The desktop relaunches mid remote update. Every boot attempt fails
//      with `update-in-progress` ("Remote Hermes update process <pid> is
//      still running; SSH startup is paused.") for ~14 minutes.
//   2. The renderer's bounded boot retry (BOOT_RETRY_MAX_ATTEMPTS, ~1 min of
//      backoff) exhausts long before the update finishes; failDesktopBoot()
//      latches the boot-failure overlay.
//   3. The remote update finally clears; main's next startHermes cycle
//      connects, logs `Remote Hermes backend is ready`, and pushes
//      `phase: 'backend.ready', error: null` over boot-progress.
//   4. The onBootProgress handler IGNORES non-error payloads once the boot
//      overlay is latched visible (it exists to swallow post-boot re-emits),
//      so nothing ever re-runs boot(). The window sits on "Connecting…" /
//      the failure overlay until a manual quit — with main-side desktop.log
//      showing zero events in between.
//
// The fix: a `backend.ready` progress event with no error, received while the
// boot overlay is still visible and no boot/switch is running, is main's
// authoritative "the blocker is gone" signal — the renderer must re-drive
// boot() off it instead of swallowing it.

type Listener = (ev: unknown) => void

class FakeWebSocket {
  static OPEN = 1
  static CLOSED = 3
  static mode: 'open' | 'fail' = 'open'
  static instances: FakeWebSocket[] = []

  readyState = 0
  private listeners: Record<string, Set<Listener>> = {}

  constructor(public url: string) {
    FakeWebSocket.instances.push(this)
    const willOpen = FakeWebSocket.mode === 'open'
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

  send(_data: string) {}

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

function fakeDesktop() {
  let bootProgressHandler: ((payload: Record<string, unknown>) => void) | null = null

  return {
    getConnection: vi.fn(async () => primaryConn),
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
    emitBootProgress(payload: Record<string, unknown>) {
      bootProgressHandler?.(payload)
    },
    onBackendExit: vi.fn(() => () => undefined),
    onConnectionApplied: vi.fn(() => () => undefined),
    onPowerResume: vi.fn(() => () => undefined),
    revalidateConnection: vi.fn(async () => ({ ok: true, rebuilt: false })),
    onWindowStateChanged: vi.fn(() => () => undefined),
    touchBackend: vi.fn(async () => undefined),
    profile: { get: vi.fn(async () => ({ profile: 'default' })) }
  }
}

function Harness() {
  useGatewayBoot({
    beforeConnectionSwitch: () => undefined,
    handleGatewayEvent: () => undefined,
    onConnectionReady: () => undefined,
    onGatewayReady: () => undefined,
    refreshHermesConfig: async () => undefined,
    refreshSessions: async () => undefined
  })

  return null
}

const originalWebSocket = globalThis.WebSocket

beforeEach(() => {
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
  vi.useFakeTimers()
  FakeWebSocket.mode = 'open'
  FakeWebSocket.instances = []
  vi.mocked(notifyError).mockReset()
  ;(globalThis as { WebSocket: unknown }).WebSocket = FakeWebSocket
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
  _resetConnectionsForTests()
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

describe('#96743 — late backend.ready must unstick a boot whose retries were exhausted', () => {
  it('re-drives boot() when backend.ready arrives after the retry budget ran out during a remote update', async () => {
    // Phase 1: every getConnection() rejects with the update-in-progress
    // refusal, exactly as main answers while the remote updater holds the
    // marker. Main-side boot progress keeps reporting retryable failures.
    const updateError = new Error('Remote Hermes update process 4242 is still running; SSH startup is paused.')
    const desktop = fakeDesktop()
    // The remote update blocks every dial until the test flips the switch —
    // mirrors the ~14 min marker pause in the issue, regardless of how many
    // boot attempts the renderer makes inside the fake-timer window.
    let updateFinished = false
    desktop.getConnection = vi.fn(async () => {
      if (!updateFinished) {
        throw updateError
      }

      return primaryConn
    })
    // bootFailureIsRetryable() consults this snapshot: main tags remote
    // update-in-progress failures retryable=true (backend-start-failure.ts:
    // everything remote that is not reauth/host-key-change is transient).
    desktop.getBootProgress = vi.fn(async () => ({
      error: updateError.message,
      fakeMode: false,
      message: `Desktop boot failed: ${updateError.message}`,
      phase: 'backend.error',
      progress: 24,
      retryable: true,
      running: false,
      timestamp: Date.now()
    }))
    ;(window as { hermesDesktop?: unknown }).hermesDesktop = desktop

    render(<Harness />)
    await flushAsync()

    const dialsAfterInitial = vi.mocked(desktop.getConnection).mock.calls.length

    // Exhaust the bounded retry budget: BOOT_RETRY_MAX_ATTEMPTS (5) retries,
    // full-jitter backoff — advance well past each draw cap (2s,4s,8s,16s,32s).
    await act(async () => {
      await vi.advanceTimersByTimeAsync(500_000)
    })

    const boot = $desktopBoot.get()

    // The failure state reached after exhaustion: overlay latched with the
    // update-in-progress message, or still cycling retries — either is a
    // valid starting point for the fix (the user's 9-minute window covers
    // both). What matters is that no dial succeeds and the UI never unlocks.
    expect(vi.mocked(desktop.getConnection).mock.calls.length).toBeGreaterThan(dialsAfterInitial)
    expect($gatewayState.get()).not.toBe('open')
    expect(boot.visible).toBe(true)

    // Phase 2: the remote update finishes. Main re-runs startHermes(),
    // connects, logs `Remote Hermes backend is ready`, and pushes the
    // backend.ready progress event. The boot snapshot now reports the healthy
    // state (main updates it on each progress publish).
    updateFinished = true
    desktop.getBootProgress = vi.fn(async () => ({
      error: null,
      fakeMode: false,
      message: 'Remote Hermes backend is ready',
      phase: 'backend.ready',
      progress: 94,
      retryable: false,
      running: true,
      timestamp: Date.now()
    }))

    await act(async () => {
      desktop.emitBootProgress({
        attemptedRemote: true,
        error: null,
        message: 'Remote Hermes backend is ready',
        phase: 'backend.ready',
        progress: 94,
        running: true,
        timestamp: Date.now()
      })
    })

    // The single backend.ready event must re-drive the whole boot pipeline:
    // a fresh dial (now succeeding), WS connect, config/session sync, and the
    // overlay dismissed — exactly what manual relaunch achieved for the
    // reporter. Fake timers: give the async boot chain room to settle.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1_000)
    })

    expect(vi.mocked(desktop.getConnection).mock.calls.length).toBeGreaterThan(dialsAfterInitial + 5)
    expect(FakeWebSocket.instances.length).toBeGreaterThan(0)
    expect($gatewayState.get()).toBe('open')
    expect($desktopBoot.get().visible).toBe(false)
    expect($desktopBoot.get().error).toBeNull()
  }, 30_000)

  it('does NOT re-drive boot() while a boot retry is still scheduled (would double-dial)', async () => {
    const updateError = new Error('Remote Hermes update process 4242 is still running; SSH startup is paused.')
    const desktop = fakeDesktop()
    let updateFinished = false
    desktop.getConnection = vi.fn(async () => {
      if (!updateFinished) {
        throw updateError
      }

      return primaryConn
    })
    desktop.getBootProgress = vi.fn(async () => ({
      error: updateError.message,
      fakeMode: false,
      message: `Desktop boot failed: ${updateError.message}`,
      phase: 'backend.error',
      progress: 24,
      retryable: true,
      running: false,
      timestamp: Date.now()
    }))
    ;(window as { hermesDesktop?: unknown }).hermesDesktop = desktop

    render(<Harness />)
    await flushAsync()

    // First failure, first retry SCHEDULED but not yet fired. While the retry
    // timer is pending, a stray backend.ready (e.g. a re-emit for the same
    // startHermes cycle) must not start a parallel boot: the pending retry
    // owns recovery and any extra dial just burns SSH reconnection cost.
    updateFinished = true
    await act(async () => {
      desktop.emitBootProgress({
        attemptedRemote: true,
        error: null,
        message: 'Remote Hermes backend is ready',
        phase: 'backend.ready',
        progress: 94,
        running: true,
        timestamp: Date.now()
      })
    })

    const dialsBeforeRetryFires = vi.mocked(desktop.getConnection).mock.calls.length

    // The scheduled retry, when it fires, is the SINGLE recovery path.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(10_000)
    })

    expect(vi.mocked(desktop.getConnection).mock.calls.length).toBe(dialsBeforeRetryFires + 1)
    expect($gatewayState.get()).toBe('open')
    expect($desktopBoot.get().visible).toBe(false)
  }, 30_000)

  it('ignores a backend.ready that arrives AFTER a completed boot (post-boot progress re-emit)', async () => {
    const desktop = fakeDesktop()

    ;(window as { hermesDesktop?: unknown }).hermesDesktop = desktop

    render(<Harness />)
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1_000)
    })

    // Healthy boot completed.
    expect($gatewayState.get()).toBe('open')
    expect($desktopBoot.get().visible).toBe(false)
    const dialsAfterHealthyBoot = vi.mocked(desktop.getConnection).mock.calls.length

    // A late re-emit (sleep/wake revalidation, HMR, etc.) must NOT start a
    // second boot — the running app owns the socket now.
    await act(async () => {
      desktop.emitBootProgress({
        attemptedRemote: true,
        error: null,
        message: 'Remote Hermes backend is ready',
        phase: 'backend.ready',
        progress: 94,
        running: true,
        timestamp: Date.now()
      })
      await vi.advanceTimersByTimeAsync(1_000)
    })

    expect(vi.mocked(desktop.getConnection).mock.calls.length).toBe(dialsAfterHealthyBoot)
    expect($gatewayState.get()).toBe('open')
  }, 30_000)
})
