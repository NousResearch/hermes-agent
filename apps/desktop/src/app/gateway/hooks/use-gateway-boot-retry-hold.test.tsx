import { act, cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { $desktopBoot } from '@/store/boot'
import { closeSecondaryGateways } from '@/store/gateway'
import { $activeGatewayProfile } from '@/store/profile'
import { $connection, $gatewayState } from '@/store/session'

import { FakeWebSocket } from '../../../test/fake-gateway-socket'

import { takeGatewaySurvivor } from './gateway-hmr-survivor'
import { useGatewayBoot } from './use-gateway-boot'

// The cold-boot bounded retry (#82679) against a remote that stays down
// (#123225). Shares the fake socket with use-gateway-boot.test.tsx; the
// desktop bridge here carries only what the boot hook touches.

vi.mock(import('@/store/notifications'), async importOriginal => ({
  ...(await importOriginal()),
  notifyError: vi.fn()
}))

vi.mock(import('@/store/terminal-backend-warning'), () => ({
  warnIfTerminalBackendUnavailable: vi.fn(async () => false)
}))

const UNREACHABLE = 'Could not reach the remote Hermes gateway while refreshing its WebSocket ticket. Try reconnecting.'

const failedRemoteBoot = {
  error: UNREACHABLE,
  fakeMode: false,
  message: `Desktop boot failed: ${UNREACHABLE}`,
  phase: 'backend.error',
  progress: 8,
  retryable: true,
  running: false
}

function fakeDesktop() {
  let bootProgressHandler: ((payload: Record<string, unknown>) => void) | null = null
  const unsubscribe = () => () => undefined

  const emitBootProgress = (payload: Record<string, unknown>) => bootProgressHandler?.(payload)

  return {
    // Main's order on every attempt: the boot's progress, its retryable
    // failure, then the rejected dial.
    getConnection: vi.fn(async (_profile?: string, _opts?: { redial?: boolean }) => {
      emitBootProgress({
        error: null,
        fakeMode: false,
        message: 'Resolving Hermes backend',
        phase: 'backend.resolve',
        progress: 8,
        running: true,
        timestamp: Date.now()
      })
      emitBootProgress({ ...failedRemoteBoot, timestamp: Date.now() })

      throw new Error(UNREACHABLE)
    }),
    getGatewayWsUrl: vi.fn(async () => 'wss://vps.example.com/api/ws'),
    // The startup boot main runs before the renderer mounts failed the same way.
    getBootProgress: vi.fn(async () => ({ ...failedRemoteBoot, timestamp: Date.now() })),
    onBootProgress: vi.fn(callback => {
      bootProgressHandler = callback

      return () => {
        bootProgressHandler = null
      }
    }),
    onBackendExit: vi.fn(unsubscribe),
    onConnectionApplied: vi.fn(unsubscribe),
    onPowerResume: vi.fn(unsubscribe),
    revalidateConnection: vi.fn(async () => ({ ok: true, rebuilt: false })),
    onWindowStateChanged: vi.fn(unsubscribe),
    touchBackend: vi.fn(async () => undefined),
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

const originalWebSocket = globalThis.WebSocket

beforeEach(() => {
  closeSurvivor()
  closeSecondaryGateways()
  $activeGatewayProfile.set('default')
  $connection.set(null)
  vi.useFakeTimers()
  FakeWebSocket.mode = 'open'
  FakeWebSocket.instances = []
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
  $connection.set(null)
  vi.useRealTimers()
  ;(globalThis as { WebSocket: unknown }).WebSocket = originalWebSocket
  delete (window as { hermesDesktop?: unknown }).hermesDesktop
})

async function advanceBackoff() {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(15_000)
  })
}

it('FIX #123225: a remote that stays down keeps the retry surface up, then lands on one stable recovery overlay', async () => {
  const desktop = fakeDesktop()
  // The dial attempt each recovery-overlay state appeared during.
  const overlayAtAttempt: number[] = []

  const offBoot = $desktopBoot.listen(state => {
    // BootFailureOverlay renders on `error && !running`.
    if (state.error && !state.running) {
      overlayAtAttempt.push(desktop.getConnection.mock.calls.length)
    }
  })

  ;(window as { hermesDesktop?: unknown }).hermesDesktop = desktop

  try {
    render(<Harness />)

    for (let i = 0; i < 7; i += 1) {
      await advanceBackoff()
    }

    // 1 initial attempt + 5 bounded retries.
    expect(desktop.getConnection).toHaveBeenCalledTimes(6)

    // A retryable failure the retry will re-attempt never flashed the
    // recovery overlay; only the last attempt concluded the boot.
    expect(overlayAtAttempt.length).toBeGreaterThan(0)
    expect(overlayAtAttempt.every(attempt => attempt === 6)).toBe(true)
    expect($desktopBoot.get().error).toBeTruthy()
    expect($desktopBoot.get().running).toBe(false)

    // Every attempt is the boot hook's own redial: main re-drives it while
    // answering background callers from its retry window.
    expect(desktop.getConnection.mock.calls.every(([, opts]) => opts?.redial === true)).toBe(true)
  } finally {
    offBoot()
  }
})
