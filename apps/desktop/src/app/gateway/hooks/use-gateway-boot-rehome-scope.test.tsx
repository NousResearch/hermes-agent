import { act, cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { $apiRequestScope } from '@/api/client'
import { $desktopBoot } from '@/store/boot'
import { closeSecondaryGateways } from '@/store/gateway'
import { $activeGatewayProfile } from '@/store/profile'
import { $connection, $gatewayState } from '@/store/session'
import { startUpdatePoller, stopUpdatePoller } from '@/store/updates'

import { deferred } from '../../../test/deferred'
import { FakeWebSocket } from '../../../test/fake-gateway-socket'

import { takeGatewaySurvivor } from './gateway-hmr-survivor'
import { useGatewayBoot } from './use-gateway-boot'

// A Settings gateway apply re-homes the window primary to another machine. A
// profile belongs to ONE machine, so the ambient request scope must move from
// (local, alpha) to (box, default) as one pair: while the new socket's ticket
// mint was in flight the scope named (box, alpha), and the version poller asked
// Box for a profile Box does not have.

vi.mock(import('@/store/notifications'), async importOriginal => ({
  ...(await importOriginal()),
  notifyError: vi.fn()
}))

vi.mock(import('@/store/terminal-backend-warning'), () => ({
  warnIfTerminalBackendUnavailable: vi.fn(async () => false)
}))

const localAlpha = {
  authMode: 'token' as const,
  baseUrl: 'http://127.0.0.1:9119',
  connectionId: 'local',
  mode: 'local' as const,
  profile: 'alpha',
  token: 'l',
  wsUrl: 'ws://127.0.0.1:9119/api/ws?token=l'
}

const boxDefault = {
  authMode: 'token' as const,
  baseUrl: 'https://box.example.com',
  connectionId: 'box',
  mode: 'remote' as const,
  profile: 'default',
  token: 'b',
  wsUrl: 'wss://box.example.com/api/ws?token=b'
}

let connectionApplied: null | (() => void) = null

function fakeDesktop() {
  const unsubscribe = () => () => undefined

  return {
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
    getConnection: vi.fn(async (): Promise<typeof boxDefault | typeof localAlpha> => localAlpha),
    getGatewayWsUrl: vi.fn(async (conn?: { wsUrl?: string }) => conn?.wsUrl ?? localAlpha.wsUrl),
    getVersion: vi.fn(async () => ({ appVersion: 'test' })),
    onBackendExit: vi.fn(unsubscribe),
    onBootProgress: vi.fn(unsubscribe),
    onConnectionApplied: vi.fn((callback: () => void) => {
      connectionApplied = callback

      return () => {
        connectionApplied = null
      }
    }),
    onPowerResume: vi.fn(unsubscribe),
    onWindowStateChanged: vi.fn(unsubscribe),
    profile: { get: vi.fn(async () => ({ profile: 'alpha' })) },
    revalidateConnection: vi.fn(async () => ({ ok: true, rebuilt: false })),
    updates: { check: vi.fn(async () => ({ supported: false })), onProgress: vi.fn(unsubscribe) }
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
  stopUpdatePoller()
  cleanup()
  closeSurvivor()
  closeSecondaryGateways()
  $connection.set(null)
  vi.useRealTimers()
  ;(globalThis as { WebSocket: unknown }).WebSocket = originalWebSocket
  delete (window as { hermesDesktop?: unknown }).hermesDesktop
})

async function flushAsync() {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(0)
  })
}

it('a Settings re-home publishes (box, default) as one pair while the new ticket is in flight', async () => {
  const desktop = fakeDesktop()

  ;(window as { hermesDesktop?: unknown }).hermesDesktop = desktop
  render(<Harness />)
  await flushAsync()
  expect($apiRequestScope.get()).toEqual({ connectionId: 'local', profile: 'alpha' })

  startUpdatePoller()
  await flushAsync()
  desktop.getVersion.mockClear()

  const seen: string[] = []
  const unlisten = $apiRequestScope.listen(({ connectionId, profile }) => seen.push(`${connectionId}|${profile}`))
  const ticket = deferred<string>()

  // Box's own remembered profile is `default`; main's legacy pick still says alpha.
  desktop.getConnection.mockResolvedValue(boxDefault)
  desktop.profile.get.mockResolvedValue({ profile: 'default' })
  desktop.getGatewayWsUrl.mockReturnValue(ticket.promise)

  try {
    act(() => connectionApplied?.())
    await flushAsync()

    // Mid-switch: the new socket is not open yet.
    expect($apiRequestScope.get()).toEqual({ connectionId: 'box', profile: 'default' })

    ticket.resolve(boxDefault.wsUrl)
    await flushAsync()

    expect($apiRequestScope.get()).toEqual({ connectionId: 'box', profile: 'default' })
    expect(seen).not.toContain('box|alpha')
    expect(desktop.getVersion).not.toHaveBeenCalledWith(
      expect.objectContaining({ connectionId: 'box', profile: 'alpha' })
    )
    expect(desktop.getVersion).toHaveBeenCalledWith(
      expect.objectContaining({ connectionId: 'box', profile: 'default' })
    )
  } finally {
    unlisten()
  }
})
