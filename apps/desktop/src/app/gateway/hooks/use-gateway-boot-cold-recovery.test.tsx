import { act, cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { $desktopBoot } from '@/store/boot'
import {
  $connectionsRegistry,
  _resetConnectionsForTests,
  selectConnection,
  setConnectionsRegistry
} from '@/store/connections'
import { closeSecondaryGateways } from '@/store/gateway'
import { endGatewaySwitch } from '@/store/gateway-switch'
import { notifyError } from '@/store/notifications'
import { $activeGatewayProfile } from '@/store/profile'
import { $connection, $gatewayState } from '@/store/session'

import { deferred } from '../../../test/deferred'
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

const local = {
  baseUrl: 'http://127.0.0.1:9191',
  connectionId: 'local',
  mode: 'local' as const,
  profile: 'default',
  wsUrl: 'ws://127.0.0.1:9191/api/ws'
}

const cloud = {
  ...local,
  baseUrl: 'https://offline.agents.nousresearch.com',
  connectionId: 'cloud',
  mode: 'remote' as const,
  wsUrl: 'wss://offline.agents.nousresearch.com/api/ws'
}

const originalWebSocket = globalThis.WebSocket
const originalDesktop = window.hermesDesktop

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

async function flushAsync() {
  await act(async () => vi.advanceTimersByTimeAsync(0))
}

beforeEach(() => {
  takeGatewaySurvivor()?.gateway.close()
  closeSecondaryGateways()
  vi.useFakeTimers()
  FakeWebSocket.mode = 'open'
  FakeWebSocket.instances = []
  FakeWebSocket.pingMode = 'pong'
  Object.assign(globalThis, { WebSocket: FakeWebSocket })
  $activeGatewayProfile.set('default')
  $connection.set(null)
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
  vi.mocked(notifyError).mockReset()
})

afterEach(() => {
  cleanup()
  takeGatewaySurvivor()?.gateway.close()
  closeSecondaryGateways()
  endGatewaySwitch()
  _resetConnectionsForTests()
  $connectionsRegistry.set(null)
  $connection.set(null)
  Object.assign(globalThis, { WebSocket: originalWebSocket })
  Object.assign(window, { hermesDesktop: originalDesktop })
  vi.useRealTimers()
})

it.each([
  ['apply', 'reject'],
  ['apply', 'resolve'],
  ['registry', 'reject'],
  ['registry', 'resolve']
] as const)('keeps a %s recovery connection when the old cold boot later %ss', async (route, settlement) => {
  const pending = deferred<typeof cloud>()
  let apply: (() => void) | undefined

  const desktop = {
    api: vi.fn(async ({ path }: { path: string }) =>
      path === '/api/profiles/active' ? { current: 'default' } : { profiles: [] }
    ),
    getConnection: vi
      .fn()
      .mockImplementationOnce(() => pending.promise)
      .mockResolvedValue(local),
    getConnectionFor: vi.fn(async () => ({ ...local, registryScoped: true })),
    getGatewayWsUrlFor: vi.fn(async () => local.wsUrl),
    getGatewayWsUrl: vi.fn(async (conn: typeof local) => conn.wsUrl),
    getBootProgress: vi.fn(async () => ({ phase: 'backend.remote', running: true, progress: 24, error: null })),
    onBootProgress: vi.fn(() => () => undefined),
    onBackendExit: vi.fn(() => () => undefined),
    onConnectionApplied: vi.fn((callback: () => void) => {
      apply = callback

      return () => undefined
    }),
    onPowerResume: vi.fn(() => () => undefined),
    onWindowStateChanged: vi.fn(() => () => undefined),
    profile: { get: vi.fn(async () => ({ profile: 'default' })) }
  }

  Object.assign(window, { hermesDesktop: desktop })
  render(<Harness />)
  await flushAsync()
  expect(FakeWebSocket.instances).toHaveLength(0)

  // User reached Gateway settings before the cloud's readiness wait settled.
  if (route === 'registry') {
    setConnectionsRegistry({
      version: 2,
      primary: 'cloud',
      secureTokenStorage: true,
      connections: [{ id: 'local', kind: 'local', label: 'This device', tokenPreview: null, tokenSet: false }]
    })
    let switching: Promise<void>

    act(() => {
      switching = selectConnection('local')
    })
    await flushAsync()
    await act(async () => switching)
  } else {
    act(() => apply?.())
    await flushAsync()
  }

  expect($connection.get()?.connectionId).toBe('local')
  expect($gatewayState.get()).toBe('open')
  expect($desktopBoot.get().error).toBeNull()

  await act(async () => {
    if (settlement === 'reject') {
      pending.reject(new Error('old cloud unreachable'))
    } else {
      pending.resolve(cloud)
    }

    await vi.advanceTimersByTimeAsync(0)
  })

  expect($connection.get()?.connectionId).toBe('local')
  expect($gatewayState.get()).toBe('open')
  expect($desktopBoot.get().error).toBeNull()
  expect(notifyError).not.toHaveBeenCalled()
  expect(FakeWebSocket.instances).toHaveLength(1)
})
