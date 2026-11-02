import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import type { HermesConnection } from '@/global'
import { TimeoutError } from '@/lib/with-timeout'
import { deferred } from '@/test/deferred'
import { FakeWebSocket } from '@/test/fake-gateway-socket'

vi.mock('@/store/session', () => ({ setConnection: vi.fn(), setGatewayState: vi.fn() }))
vi.mock('@/store/notify-baseline', () => ({ markNativeNotifyBaseline: vi.fn() }))
vi.mock('@/store/session-states', () => ({
  reconcileBusyStatesOnReconnect: vi.fn(),
  resetRouteOwnedTileRuntimeBindings: vi.fn(),
  resetTileRuntimeBindings: vi.fn()
}))

const { $connectionsRegistry } = await import('./connection-registry-state')

const {
  closeSecondaryGateways,
  configureGatewayRegistry,
  disposeSecondariesForConnection,
  liveSecondaryConnectionIds,
  openGatewayForAgent,
  requestGatewayForAgent,
  setPrimaryGateway,
  setPrimaryGatewayConnection
} = await import('./gateway')

const { startPersistentGateways } = await import('./gateway-persistence')
const { onKeptRouteOpened } = await import('./gateway-kept-routes')

function descriptor(): HermesConnection {
  return {
    connectionId: 'local',
    profile: 'default',
    mode: 'local',
    authMode: 'token',
    token: 'test',
    wsUrl: 'ws://local.invalid/api/ws',
    baseUrl: 'http://local.invalid',
    registryScoped: true,
    isFullscreen: false,
    nativeOverlayWidth: 0,
    logs: [],
    windowButtonPosition: null
  }
}

const getConnectionFor = vi.fn(async () => descriptor())
let stop: () => void

beforeEach(() => {
  vi.useFakeTimers()
  vi.stubGlobal('WebSocket', FakeWebSocket)
  FakeWebSocket.mode = 'open'
  FakeWebSocket.pingMode = 'pong'
  FakeWebSocket.instances = []
  getConnectionFor.mockReset().mockResolvedValue(descriptor())
  configureGatewayRegistry({ onEvent: vi.fn() })
  setPrimaryGateway({ connectionState: 'open' } as never, 'default')
  setPrimaryGatewayConnection({ connectionId: 'remote-primary', mode: 'remote' })
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: {
      probeLocalBackend: vi.fn(async () => ({ bootstrapNeeded: false })),
      getConnectionFor,
      getGatewayWsUrlFor: vi.fn(async () => descriptor().wsUrl)
    }
  })
  $connectionsRegistry.set({
    version: 2,
    primary: 'remote-primary',
    secureTokenStorage: true,
    connections: [
      { id: 'remote-primary', kind: 'remote', label: 'Remote', tokenSet: true, tokenPreview: null },
      { id: 'local', kind: 'local', label: 'This device', tokenSet: false, tokenPreview: null }
    ]
  })
  stop = () => undefined
})

afterEach(() => {
  stop()
  closeSecondaryGateways()
  $connectionsRegistry.set(null)
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
  vi.useRealTimers()
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
})

const permanentErrors = [
  'Gateway incompatible (runtime_protocol)',
  'hermes gateway ensure produced no result (exit 2)',
  'Invalid local gateway endpoint',
  'Unsafe gateway control path'
]

it.each(permanentErrors)('parks an initial permanent failure until an explicit retry: %s', async message => {
  getConnectionFor.mockRejectedValue(new Error(message))
  stop = startPersistentGateways()
  await vi.advanceTimersByTimeAsync(0)
  expect(getConnectionFor).toHaveBeenCalledTimes(1)

  await vi.advanceTimersByTimeAsync(180_000)
  expect(getConnectionFor).toHaveBeenCalledTimes(1)
  expect(liveSecondaryConnectionIds().has('local')).toBe(false)

  await expect(requestGatewayForAgent('local', 'default', 'session.control.read')).rejects.toThrow()
  expect(getConnectionFor).toHaveBeenCalledTimes(1)

  getConnectionFor.mockResolvedValue(descriptor())
  const retry = openGatewayForAgent('local', 'default', { spawnPriority: 'foreground' })
  await vi.advanceTimersByTimeAsync(0)
  await retry
  expect(liveSecondaryConnectionIds().has('local')).toBe(true)
  expect(getConnectionFor).toHaveBeenCalledTimes(2)
})

it('does not rearm a permanent failure parked by socket reconnect, but a connection edit can recover it', async () => {
  stop = startPersistentGateways()
  await vi.advanceTimersByTimeAsync(0)
  expect(liveSecondaryConnectionIds().has('local')).toBe(true)

  getConnectionFor.mockRejectedValue(new Error('Gateway incompatible (runtime_protocol)'))
  FakeWebSocket.instances[0].drop()
  await vi.advanceTimersByTimeAsync(20_000)
  expect(getConnectionFor).toHaveBeenCalledTimes(2)
  await vi.advanceTimersByTimeAsync(180_000)
  expect(getConnectionFor).toHaveBeenCalledTimes(2)

  getConnectionFor.mockResolvedValue(descriptor())
  disposeSecondariesForConnection('local', { redial: true })
  await vi.advanceTimersByTimeAsync(0)
  expect(liveSecondaryConnectionIds().has('local')).toBe(true)
  expect(getConnectionFor).toHaveBeenCalledTimes(3)
})

it('recovers transient startup and socket failures automatically and refreshes kept-route listeners', async () => {
  const opened = vi.fn()
  const off = onKeptRouteOpened(opened)
  getConnectionFor.mockRejectedValueOnce(new Error('ECONNREFUSED'))
  stop = startPersistentGateways()

  try {
    await vi.advanceTimersByTimeAsync(20_000)
    expect(liveSecondaryConnectionIds().has('local')).toBe(true)
    expect(getConnectionFor).toHaveBeenCalledTimes(2)
    expect(opened).toHaveBeenCalledTimes(1)

    getConnectionFor.mockRejectedValueOnce(new Error('ECONNRESET'))
    FakeWebSocket.instances[0].drop()
    await vi.advanceTimersByTimeAsync(20_000)
    expect(liveSecondaryConnectionIds().has('local')).toBe(true)
    expect(getConnectionFor).toHaveBeenCalledTimes(4)
    expect(opened).toHaveBeenCalledTimes(2)
  } finally {
    off()
  }
})

it('preserves the stalled-dial budget across keeper ticks and lets a foreground open recover', async () => {
  getConnectionFor.mockRejectedValue(new TimeoutError('Timed out connecting to profile "default"'))
  stop = startPersistentGateways()
  await vi.advanceTimersByTimeAsync(180_000)
  expect(getConnectionFor).toHaveBeenCalledTimes(3)
  await vi.advanceTimersByTimeAsync(180_000)
  expect(getConnectionFor).toHaveBeenCalledTimes(3)

  getConnectionFor.mockResolvedValue(descriptor())
  const retry = openGatewayForAgent('local', 'default', { spawnPriority: 'foreground' })
  await vi.advanceTimersByTimeAsync(0)
  await retry
  expect(liveSecondaryConnectionIds().has('local')).toBe(true)
})

it('coalesces overlapping maintenance and never resurrects a removed gateway when its dial finishes', async () => {
  const pending = deferred<HermesConnection>()
  getConnectionFor.mockReturnValueOnce(pending.promise)
  stop = startPersistentGateways()
  await vi.advanceTimersByTimeAsync(0)
  expect(getConnectionFor).toHaveBeenCalledTimes(1)

  $connectionsRegistry.set({ ...$connectionsRegistry.get()! })
  await vi.advanceTimersByTimeAsync(60_000)
  expect(getConnectionFor).toHaveBeenCalledTimes(1)

  disposeSecondariesForConnection('local')
  const registry = $connectionsRegistry.get()!
  $connectionsRegistry.set({ ...registry, connections: registry.connections.filter(c => c.id !== 'local') })
  pending.resolve(descriptor())
  await vi.advanceTimersByTimeAsync(180_000)
  expect(liveSecondaryConnectionIds().has('local')).toBe(false)
  expect(getConnectionFor).toHaveBeenCalledTimes(1)
  expect(FakeWebSocket.instances.every(socket => socket.readyState === FakeWebSocket.CLOSED)).toBe(true)
})
