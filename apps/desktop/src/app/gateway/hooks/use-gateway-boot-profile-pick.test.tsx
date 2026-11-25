import { act, cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { $desktopBoot } from '@/store/boot'
import { closeSecondaryGateways, isActivePrimary } from '@/store/gateway'
import { $activeGatewayProfile, ensureGatewayProfile } from '@/store/profile'
import { $connection, $gatewayState } from '@/store/session'

import { deferred } from '../../../test/deferred'
import { FakeWebSocket } from '../../../test/fake-gateway-socket'

import { takeGatewaySurvivor } from './gateway-hmr-survivor'
import { useGatewayBoot } from './use-gateway-boot'

// The blocked update-hold screen's renderer leg (R8 D3/M6): a hold the main
// process publishes on boot progress lands in $desktopBoot even after boot.
// Shares the fake socket with use-gateway-boot.test.tsx; the desktop bridge
// here carries only what the boot hook touches.

vi.mock(import('@/store/notifications'), async importOriginal => ({
  ...(await importOriginal()),
  notifyError: vi.fn()
}))

vi.mock(import('@/store/terminal-backend-warning'), () => ({
  warnIfTerminalBackendUnavailable: vi.fn(async () => false)
}))

const primaryConn = {
  authMode: 'token' as const,
  baseUrl: 'https://vps.example.com',
  connectionId: 'primary-vps',
  profile: 'default',
  token: 't',
  wsUrl: 'wss://vps.example.com/api/ws?token=t'
}

const coderConn = {
  ...primaryConn,
  baseUrl: 'https://coder.example.com',
  connectionId: 'coder-remote',
  profile: 'coder',
  token: 'c',
  wsUrl: 'wss://coder.example.com/api/ws?token=c'
}

function fakeDesktop() {
  let bootProgressHandler: ((payload: Record<string, unknown>) => void) | null = null
  const unsubscribe = () => () => undefined

  return {
    getConnection: vi.fn(async (_profile?: null | string): Promise<typeof primaryConn> => primaryConn),
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

function Harness({ refreshSessions }: { refreshSessions: () => Promise<void> }) {
  useGatewayBoot({
    beforeConnectionSwitch: () => undefined,
    handleGatewayEvent: () => undefined,
    handleServerRequest: () => false,
    onConnectionReady: () => undefined,
    onGatewayReady: () => undefined,
    refreshHermesConfig: async () => undefined,
    refreshSessions
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
  cleanup()
  closeSurvivor()
  closeSecondaryGateways()
  $connection.set(null)
  vi.useRealTimers()
  ;(globalThis as { WebSocket: unknown }).WebSocket = originalWebSocket
  delete (window as { hermesDesktop?: unknown }).hermesDesktop
})

// T19 (#134889): the first secondary-profile click right after boot lands
// while boot is still resolving the primary. Boot's late adoption must not
// re-point the active route (and the session list it scopes) at default.
it('a profile picked while boot is still connecting survives the primary adoption', async () => {
  const desktop = fakeDesktop()
  const primaryLookup = deferred<typeof primaryConn>()
  desktop.getConnection.mockImplementation(async (profile?: null | string) =>
    (profile ?? '').trim() === 'coder' ? coderConn : primaryLookup.promise
  )
  ;(window as { hermesDesktop?: unknown }).hermesDesktop = desktop
  const scopes: string[] = []
  render(<Harness refreshSessions={async () => void scopes.push($activeGatewayProfile.get())} />)
  await act(async () => {
    await vi.advanceTimersByTimeAsync(0)
  })

  vi.useRealTimers()
  await ensureGatewayProfile('coder')
  expect(isActivePrimary()).toBe(false)

  primaryLookup.resolve(primaryConn)
  await new Promise(resolve => setTimeout(resolve, 50))
  vi.useFakeTimers()
  await act(async () => {
    await vi.advanceTimersByTimeAsync(0)
  })

  expect($desktopBoot.get().running).toBe(false)
  expect($activeGatewayProfile.get()).toBe('coder')
  expect(isActivePrimary()).toBe(false)
  // Socket, profile and connection descriptor agree on the pick
  // (apps/desktop/AGENTS.md): boot records the primary, it doesn't publish it.
  expect($connection.get()?.connectionId).toBe('coder-remote')
  expect($connection.get()?.profile).toBe('coder')
  expect(scopes).toEqual(['coder'])
})
