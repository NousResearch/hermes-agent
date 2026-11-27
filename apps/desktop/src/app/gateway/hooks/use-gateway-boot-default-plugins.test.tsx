import { act, cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { BACKEND_BOOT_WAIT_TIMEOUT_MS } from '@/lib/with-timeout'
import { $desktopBoot } from '@/store/boot'
import { closeSecondaryGateways } from '@/store/gateway'
import { $activeGatewayProfile } from '@/store/profile'
import { $connection, $gatewayState } from '@/store/session'

import { FakeWebSocket } from '../../../test/fake-gateway-socket'

import { takeGatewaySurvivor } from './gateway-hmr-survivor'
import { useGatewayBoot } from './use-gateway-boot'

vi.mock(import('@/store/notifications'), async importOriginal => ({
  ...(await importOriginal()),
  notifyError: vi.fn()
}))

vi.mock(import('@/store/terminal-backend-warning'), () => ({
  warnIfTerminalBackendUnavailable: vi.fn(async () => false)
}))

const primaryConn = {
  authMode: 'token' as const,
  baseUrl: 'http://127.0.0.1:8642',
  connectionId: 'local',
  mode: 'local' as const,
  profile: 'default',
  token: 't',
  wsUrl: 'ws://127.0.0.1:8642/api/ws?token=t'
}

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason?: unknown) => void

  const promise = new Promise<T>((res, rej) => {
    resolve = res
    reject = rej
  })

  return { promise, reject, resolve }
}

function fakeDesktop(getConnection: () => Promise<typeof primaryConn>) {
  let bootProgressHandler: ((payload: Record<string, unknown>) => void) | null = null
  const unsubscribe = () => () => undefined

  return {
    getConnection: vi.fn(getConnection),
    getGatewayWsUrl: vi.fn(async (conn?: { wsUrl?: string }) => conn?.wsUrl ?? primaryConn.wsUrl),
    getBootProgress: vi.fn(async () => ({
      error: null,
      fakeMode: false,
      message: '',
      phase: 'init',
      progress: 0,
      retryable: false,
      running: true,
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
    onBackendExit: vi.fn(unsubscribe),
    onConnectionApplied: vi.fn(unsubscribe),
    onPowerResume: vi.fn(unsubscribe),
    revalidateConnection: vi.fn(async () => ({ ok: true, rebuilt: false })),
    onWindowStateChanged: vi.fn(unsubscribe),
    profile: { get: vi.fn(async () => ({ profile: 'default' })) }
  }
}

function Harness() {
  useGatewayBoot({
    beforeConnectionSwitch: () => undefined,
    handleGatewayEvent: () => undefined,
    handleServerRequest: () => false,
    onConnectionReady: () => undefined,
    onGatewayReady: () => undefined,
    refreshHermesConfig: async () => undefined,
    refreshSessions: async () => undefined
  })

  return null
}

function closeSurvivor() {
  try {
    takeGatewaySurvivor()?.gateway.close()
  } catch {
    // ignore
  }
}

async function flushAsync() {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(0)
  })
}

const originalWebSocket = globalThis.WebSocket

beforeEach(() => {
  closeSurvivor()
  closeSecondaryGateways()
  $activeGatewayProfile.set('default')
  $connection.set(null)
  vi.useFakeTimers()
  FakeWebSocket.mode = 'open'
  FakeWebSocket.instances = []
  FakeWebSocket.pingMode = 'pong'
  ;(globalThis as { WebSocket: unknown }).WebSocket = FakeWebSocket
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
  closeSurvivor()
  closeSecondaryGateways()
  vi.useRealTimers()
  ;(globalThis as { WebSocket: unknown }).WebSocket = originalWebSocket
  delete (window as { hermesDesktop?: unknown }).hermesDesktop
})

it('pauses the backend boot connect timer during backend.default-plugins and connects once preflight finishes', async () => {
  const pending = deferred<typeof primaryConn>()

  const desktop = fakeDesktop(() => pending.promise)

  ;(window as { hermesDesktop?: unknown }).hermesDesktop = desktop

  render(<Harness />)
  await flushAsync()

  // Consume part of the boot budget before entering backend.default-plugins.
  await act(async () => {
    await vi.advanceTimersByTimeAsync(10_000)
  })

  act(() => {
    desktop.emitBootProgress({
      error: null,
      fakeMode: false,
      message: 'Installing default plugins...',
      phase: 'backend.default-plugins',
      progress: 45,
      running: true,
      timestamp: Date.now()
    })
  })

  // Advance well past BACKEND_BOOT_WAIT_TIMEOUT_MS while default-plugins preflight runs: must NOT time out.
  await act(async () => {
    await vi.advanceTimersByTimeAsync(BACKEND_BOOT_WAIT_TIMEOUT_MS + 60_000)
  })
  expect($desktopBoot.get().error).toBeNull()
  expect($desktopBoot.get().phase).toBe('backend.default-plugins')

  // Leaving backend.default-plugins resumes the remaining connect budget; resolving connects cleanly.
  act(() => {
    desktop.emitBootProgress({
      error: null,
      fakeMode: false,
      message: 'Starting Hermes gateway',
      phase: 'backend.starting',
      progress: 70,
      running: true,
      timestamp: Date.now()
    })
  })

  await act(async () => {
    pending.resolve(primaryConn)
    await vi.advanceTimersByTimeAsync(0)
  })

  expect($desktopBoot.get().error).toBeNull()
  expect($connection.get()?.connectionId).toBe(primaryConn.connectionId)
})

it('still times out if main wedges after leaving backend.default-plugins', async () => {
  const desktop = fakeDesktop(() => new Promise<typeof primaryConn>(() => undefined))

  ;(window as { hermesDesktop?: unknown }).hermesDesktop = desktop

  render(<Harness />)
  await flushAsync()

  act(() => {
    desktop.emitBootProgress({
      error: null,
      fakeMode: false,
      message: 'Installing default plugins...',
      phase: 'backend.default-plugins',
      progress: 45,
      running: true,
      timestamp: Date.now()
    })
  })

  await act(async () => {
    await vi.advanceTimersByTimeAsync(BACKEND_BOOT_WAIT_TIMEOUT_MS + 30_000)
  })
  expect($desktopBoot.get().error).toBeNull()

  act(() => {
    desktop.emitBootProgress({
      error: null,
      fakeMode: false,
      message: 'Starting Hermes gateway',
      phase: 'backend.starting',
      progress: 70,
      running: true,
      timestamp: Date.now()
    })
  })

  await act(async () => {
    await vi.advanceTimersByTimeAsync(BACKEND_BOOT_WAIT_TIMEOUT_MS)
  })
  expect($desktopBoot.get().error).toContain('Timed out connecting to Hermes backend')
})

it('disposes the $desktopBoot subscription on unmount while waiting in backend.default-plugins', async () => {
  const desktop = fakeDesktop(() => new Promise<typeof primaryConn>(() => undefined))

  ;(window as { hermesDesktop?: unknown }).hermesDesktop = desktop

  const unsubSpies: Array<ReturnType<typeof vi.fn>> = []
  const origSubscribe = $desktopBoot.subscribe.bind($desktopBoot)

  const subscribeSpy = vi.spyOn($desktopBoot, 'subscribe').mockImplementation(listener => {
    const off = origSubscribe(listener)
    const wrapped = vi.fn(() => off())
    unsubSpies.push(wrapped)

    return wrapped
  })

  try {
    const { unmount } = render(<Harness />)
    await flushAsync()

    act(() => {
      desktop.emitBootProgress({
        error: null,
        fakeMode: false,
        message: 'Installing default plugins...',
        phase: 'backend.default-plugins',
        progress: 45,
        running: true,
        timestamp: Date.now()
      })
    })

    expect(unsubSpies.length).toBeGreaterThan(0)
    expect(unsubSpies.some(spy => spy.mock.calls.length === 0)).toBe(true)

    unmount()
    await flushAsync()

    for (const spy of unsubSpies) {
      expect(spy).toHaveBeenCalledTimes(1)
    }

    await act(async () => {
      await vi.advanceTimersByTimeAsync(BACKEND_BOOT_WAIT_TIMEOUT_MS + 10_000)
    })
    expect($desktopBoot.get().error).toBeNull()
  } finally {
    subscribeSpy.mockRestore()
  }
})
