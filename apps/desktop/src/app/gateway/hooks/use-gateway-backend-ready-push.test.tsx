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

// #96743 item 1: main pushes a `hermes:backend-ready` event the MOMENT the
// primary connection is resolved, carrying the full connection descriptor.
// The renderer's boot loop subscribes and uses the pushed descriptor as the
// connection for the in-flight boot when one is pending — instead of relying
// exclusively on its own getConnection() IPC round-trip, which can lose the
// race against main settling the promise after the renderer's retry budget
// ran out during a long remote-update gate.

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
  let backendReadyHandler: ((payload: Record<string, unknown>) => void) | null = null

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
    // Item 1's new push channel: main → renderer, fires once per successful
    // primary connect, carrying the full connection descriptor.
    onBackendReady: vi.fn(callback => {
      backendReadyHandler = callback

      return () => {
        backendReadyHandler = null
      }
    }),
    emitBackendReady(payload: Record<string, unknown>) {
      backendReadyHandler?.(payload)
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

async function flushAsync() {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(0)
  })
}

describe('#96743 item 1 — main pushes backend-ready with the resolved connection', () => {
  it('a boot stuck behind getConnection() unblocks when main pushes the resolved connection', async () => {
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

    // Retry budget exhausts — the boot overlay is latched, no further
    // attempts are scheduled. Nothing yet can recover this except main
    // pushing the resolved connection.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(500_000)
    })

    const dialsBeforePush = vi.mocked(desktop.getConnection).mock.calls.length

    expect($gatewayState.get()).not.toBe('open')
    expect($desktopBoot.get().visible).toBe(true)
    expect(dialsBeforePush).toBeGreaterThan(1)

    // Main finished the remote update, resolved the connection, and PUSHED
    // it to the renderer with the full descriptor. boot() must consume it
    // directly — without waiting on a new getConnection() round-trip.
    updateFinished = true
    await act(async () => {
      desktop.emitBackendReady({
        connection: primaryConn
      })
      await vi.advanceTimersByTimeAsync(1_000)
    })

    // Boot completed via the PUSHED connection, no extra dial. The
    // FakeWebSocket for the gateway handshake is the proof the boot
    // proceeded past getConnection all the way to gateway.connect().
    expect(FakeWebSocket.instances.length).toBeGreaterThan(0)
    expect($gatewayState.get()).toBe('open')
    expect($desktopBoot.get().visible).toBe(false)
    expect($desktopBoot.get().error).toBeNull()
    // Emphasize the point: the boot did NOT need a fresh dial to recover.
    expect(vi.mocked(desktop.getConnection).mock.calls.length).toBe(dialsBeforePush)
  }, 30_000)

  it('a dark push after a completed boot is ignored (post-boot stability)', async () => {
    const desktop = fakeDesktop()

    ;(window as { hermesDesktop?: unknown }).hermesDesktop = desktop

    render(<Harness />)
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1_000)
    })

    expect($gatewayState.get()).toBe('open')
    expect($desktopBoot.get().visible).toBe(false)
    const dialsAfterBoot = vi.mocked(desktop.getConnection).mock.calls.length

    // A late backend-ready push (e.g. a profile-handoff re-emit) must NOT
    // disrupt the healthy boot. The renderer already owns a live connection;
    // parroting an older descriptor would be a regression.
    await act(async () => {
      desktop.emitBackendReady({
        connection: {
          ...primaryConn,
          token: 'stale-token',
          wsUrl: 'wss://vps.example.com/api/ws?token=stale-token'
        }
      })
      await vi.advanceTimersByTimeAsync(1_000)
    })

    expect($gatewayState.get()).toBe('open')
    expect(vi.mocked(desktop.getConnection).mock.calls.length).toBe(dialsAfterBoot)
  }, 30_000)
})
