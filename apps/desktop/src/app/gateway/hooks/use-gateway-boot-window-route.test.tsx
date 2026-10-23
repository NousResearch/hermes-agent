import { act, cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { closeSecondaryGateways, isActivePrimary } from '@/store/gateway'
import { endGatewaySwitch } from '@/store/gateway-switch'
import { $activeGatewayProfile, $profiles, ensureGatewayAgent } from '@/store/profile'
import { $connection, $gatewayState } from '@/store/session'

import { FakeWebSocket } from '../../../test/fake-gateway-socket'

import { takeGatewaySurvivor } from './gateway-hmr-survivor'
import { useGatewayBoot } from './use-gateway-boot'

vi.mock(import('@/store/terminal-backend-warning'), () => ({
  warnIfTerminalBackendUnavailable: vi.fn(async () => false)
}))

// #143725: main's per-window route (New Window, profile-less getConnection,
// terminal panes) must follow what the window shows on every activation path,
// including the primary fast path that never re-publishes through the boot hook.

type Route = { connectionId: null | string; profile?: string; registryScoped: boolean } | null

const descriptor = (connectionId: string, profile: string) => ({
  authMode: 'token' as const,
  baseUrl: `https://${connectionId}.example.com`,
  connectionId,
  mode: 'remote' as const,
  profile,
  registryScoped: true,
  token: 't',
  wsUrl: `wss://${connectionId}.example.com/api/ws?profile=${profile}`
})

const primary = descriptor('tower', 'default')

let routes: Route[] = []

function fakeDesktop() {
  return {
    getConnection: vi.fn(async () => primary),
    getConnectionFor: vi.fn(async ({ connectionId, profile }: { connectionId: string; profile: string }) =>
      descriptor(connectionId, profile)
    ),
    getGatewayWsUrl: vi.fn(async (conn?: { wsUrl?: string }) => conn?.wsUrl ?? primary.wsUrl),
    getGatewayWsUrlFor: vi.fn(
      async ({ connectionId, profile }: { connectionId: string; profile: string }) =>
        descriptor(connectionId, profile).wsUrl
    ),
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
    onBootProgress: vi.fn(() => () => undefined),
    onBackendExit: vi.fn(() => () => undefined),
    onConnectionApplied: vi.fn(() => () => undefined),
    onPowerResume: vi.fn(() => () => undefined),
    onWindowStateChanged: vi.fn(() => () => undefined),
    revalidateConnection: vi.fn(async () => ({ ok: true, rebuilt: false })),
    profile: { get: vi.fn(async () => ({ profile: 'default' })) },
    setActiveConnectionRoute: vi.fn((route: Route) => void routes.push(route))
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

const originalWebSocket = globalThis.WebSocket

function dropSurvivor() {
  takeGatewaySurvivor()?.gateway.close()
}

beforeEach(() => {
  dropSurvivor()
  closeSecondaryGateways()
  routes = []
  $activeGatewayProfile.set('default')
  $connection.set(null)
  $profiles.set([])
  $gatewayState.set('idle')
  vi.useFakeTimers()
  FakeWebSocket.mode = 'open'
  FakeWebSocket.instances = []
  FakeWebSocket.pingMode = 'pong'
  ;(globalThis as { WebSocket: unknown }).WebSocket = FakeWebSocket
  ;(window as { hermesDesktop?: unknown }).hermesDesktop = fakeDesktop()
})

afterEach(() => {
  cleanup()
  dropSurvivor()
  closeSecondaryGateways()
  $activeGatewayProfile.set('default')
  $connection.set(null)
  endGatewaySwitch()
  vi.useRealTimers()
  ;(globalThis as { WebSocket: unknown }).WebSocket = originalWebSocket
  delete (window as { hermesDesktop?: unknown }).hermesDesktop
})

async function flushAsync() {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(0)
  })
}

async function activate(connectionId: string, profile: string) {
  await act(async () => {
    const switching = ensureGatewayAgent(connectionId, profile)
    await vi.advanceTimersByTimeAsync(0)
    await switching
  })
}

it.each([
  ['another gateway', 'laptop', 'default'],
  ['another profile on the primary gateway', 'tower', 'work']
])('returning to the primary after visiting %s routes main back to the primary', async (_, connectionId, profile) => {
  render(<Harness />)
  await flushAsync()
  expect($gatewayState.get()).toBe('open')
  expect(routes.at(-1)).toEqual({ connectionId: 'tower', profile: 'default', registryScoped: true })

  await activate(connectionId, profile)
  expect(isActivePrimary()).toBe(false)
  expect(routes.at(-1)).toEqual({ connectionId, profile, registryScoped: true })

  await activate('tower', 'default')
  expect(isActivePrimary()).toBe(true)
  expect($connection.get()).toMatchObject({ connectionId: 'tower', profile: 'default' })
  expect(routes.at(-1)).toEqual({ connectionId: 'tower', profile: 'default', registryScoped: true })
})
