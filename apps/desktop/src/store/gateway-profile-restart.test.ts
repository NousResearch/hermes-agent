import { afterEach, expect, it, vi } from 'vitest'

import type { HermesConnection } from '@/global'

const hooks = vi.hoisted(() => ({ reset: vi.fn(), busy: vi.fn(), publish: vi.fn() }))
vi.mock('@/store/session', () => ({ setConnection: hooks.publish, setGatewayState: vi.fn() }))
vi.mock('@/store/session-states', () => ({
  resetTileRuntimeBindings: hooks.reset,
  reconcileBusyStatesOnReconnect: hooks.busy
}))
vi.mock('@/store/notify-baseline', () => ({ markNativeNotifyBaseline: vi.fn() }))
vi.mock('@/hermes', () => ({
  setApiRequestConnection: vi.fn(),
  HermesGateway: class {
    connectionState = 'closed'
    url = ''
    connect = vi.fn(async (url: string) => {
      this.url = url
      this.connectionState = 'open'
    })
    close = vi.fn(() => {
      this.connectionState = 'closed'
    })
    onEvent = vi.fn(() => () => {})
    onState = vi.fn(() => () => {})
    request = vi.fn(async () => ({ servedBy: this.url }))
  }
}))

const { HermesGateway } = await import('@/hermes')
const gateway = await import('./gateway')

function connection(connectionId: string, profile: string, generation = 'old'): HermesConnection {
  return {
    connectionId,
    profile,
    mode: 'local',
    baseUrl: 'http://127.0.0.1:9876',
    token: 'fixture',
    wsUrl: `ws://127.0.0.1:9876/${connectionId}/${profile}/${generation}`,
    logs: [],
    isFullscreen: false,
    nativeOverlayWidth: 0,
    windowButtonPosition: null
  }
}

function setup() {
  gateway.configureGatewayRegistry({ onEvent: vi.fn() })
  const primary = new HermesGateway()
  gateway.setPrimaryGateway(primary, 'coder')
  gateway.setPrimaryGatewayConnectionId('local')
  window.hermesDesktop = {
    getConnection: vi.fn(async (profile: string) => connection('local', profile)),
    getConnectionFor: vi.fn(async ({ connectionId, profile }) => connection(connectionId, profile))
  } as unknown as typeof window.hermesDesktop

  return primary
}

afterEach(() => {
  gateway.closeSecondaryGateways()
  vi.clearAllMocks()
})

it('does not reconcile another primary owner when connection changes during reconnect', async () => {
  const primary = setup()
  let connected!: () => void
  vi.mocked(primary.connect).mockImplementation(
    () =>
      new Promise<void>(resolve => {
        connected = resolve
      })
  )
  const pending = gateway.reconnectGatewayForAgent('local', 'coder', connection('local', 'coder', 'restarted'))
  await vi.waitFor(() => expect(primary.connect).toHaveBeenCalledOnce())
  gateway.setPrimaryGateway(new HermesGateway(), 'coder')
  gateway.setPrimaryGatewayConnectionId('other')
  connected()

  await expect(pending).rejects.toThrow('owner changed')
  expect(hooks.busy).not.toHaveBeenCalled()
})

it('refreshes only the requested secondary from its returned descriptor before the next RPC', async () => {
  const primary = setup()
  await gateway.ensureGatewayForAgent('other', 'writer')
  const other = gateway.activeGateway()!
  await gateway.ensureGatewayForAgent('local', 'writer')
  const selected = gateway.activeGateway()!
  await gateway.ensureGatewayForAgent('other', 'writer')
  const fresh = connection('local', 'writer', 'restarted')
  vi.clearAllMocks()

  await gateway.reconnectGatewayForAgent('local', 'writer', fresh)
  expect(await gateway.requestGatewayForAgent('local', 'writer', 'plugins.list')).toEqual({ servedBy: fresh.wsUrl })
  expect(selected.close).toHaveBeenCalledOnce()
  expect(other.close).not.toHaveBeenCalled()
  expect(primary.close).not.toHaveBeenCalled()
  expect(gateway.activeGateway()).toBe(other)
  expect(hooks.publish).not.toHaveBeenCalled()
  expect(hooks.reset).toHaveBeenCalledWith({ connectionId: 'local', profile: 'writer' })
})

it('refreshes the exact primary owner without switching an unrelated foreground', async () => {
  const primary = setup()
  await gateway.ensureGatewayForAgent('other', 'coder')
  const other = gateway.activeGateway()!
  const fresh = connection('local', 'coder', 'restarted')
  vi.clearAllMocks()

  await gateway.reconnectGatewayForAgent('local', 'coder', fresh)
  expect(await gateway.requestGatewayForAgent('local', 'coder', 'plugins.list')).toEqual({ servedBy: fresh.wsUrl })
  expect(primary.close).toHaveBeenCalledOnce()
  expect(other.close).not.toHaveBeenCalled()
  expect(gateway.activeGateway()).toBe(other)
  expect(hooks.publish).not.toHaveBeenCalled()
  expect(hooks.reset).toHaveBeenCalledWith({ connectionId: 'local', profile: 'coder' })
})
