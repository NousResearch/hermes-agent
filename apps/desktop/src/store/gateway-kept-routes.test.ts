import { afterEach, beforeEach, expect, it, vi } from 'vitest'

const gatewayMocks = vi.hoisted(() => ({
  connect: vi.fn(async (_wsUrl: string): Promise<void> => undefined)
}))

vi.mock('@/hermes', () => ({
  setApiRequestConnection: vi.fn(),
  HermesGateway: class {
    connectionState = 'closed'
    connect = async (wsUrl: string): Promise<void> => {
      await gatewayMocks.connect(wsUrl)
      this.connectionState = 'open'
    }
    close = (): void => {
      this.connectionState = 'closed'
    }
    request = async (): Promise<unknown> => ({})
    onEvent = vi.fn(() => () => {})
    onState = vi.fn(() => () => {})
  }
}))
vi.mock('@/store/session', () => ({ setConnection: vi.fn(), setGatewayState: vi.fn() }))
vi.mock('@/store/notify-baseline', () => ({ markNativeNotifyBaseline: vi.fn() }))

const {
  closeSecondaryGateways,
  configureGatewayRegistry,
  disposeSecondariesForConnection,
  openGatewayForAgent,
  pruneSecondaryGateways,
  retainGatewayForRelay,
  setPrimaryGateway
} = await import('./gateway')

const { keepRoute, onKeptRouteOpened } = await import('./gateway-kept-routes')

let endpoint = 'wss://homelab.invalid/api/ws?token=old'

/** Idle long enough that the pruner's young-socket grace no longer applies. */
function pastMinLifetime(): void {
  const later = Date.now() + 10 * 60_000
  vi.spyOn(Date, 'now').mockReturnValue(later)
}

beforeEach(() => {
  endpoint = 'wss://homelab.invalid/api/ws?token=old'
  configureGatewayRegistry({ onEvent: vi.fn() })
  setPrimaryGateway({ connectionState: 'open' } as never, 'default')
  ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = {
    getConnection: vi.fn(async () => ({ mode: 'remote', wsUrl: endpoint })),
    getConnectionFor: vi.fn(async () => ({ mode: 'remote', wsUrl: endpoint }))
  }
})

afterEach(() => {
  closeSecondaryGateways()
  vi.restoreAllMocks()
  vi.clearAllMocks()
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
})

it('a kept gateway survives idle pruning, yet an edit to it redials onto the new endpoint at once', async () => {
  const release = keepRoute('homelab', 'default')
  const opened = vi.fn()
  const off = onKeptRouteOpened(opened)

  await openGatewayForAgent('homelab', 'default')
  pastMinLifetime()
  pruneSecondaryGateways(new Set())
  expect(gatewayMocks.connect).toHaveBeenCalledTimes(1)

  // The user edits the gateway's token: the keep pin must not hold the old socket.
  endpoint = 'wss://homelab.invalid/api/ws?token=new'
  disposeSecondariesForConnection('homelab', { redial: true })
  await vi.waitFor(() => expect(gatewayMocks.connect).toHaveBeenLastCalledWith(endpoint))

  // Both opens (first dial and the redial) announce themselves, so lists re-pull.
  expect(opened).toHaveBeenCalledTimes(2)
  off()
  release()
})

it('a relay lease still defers an edit on a kept gateway until it is released; the keep pin never does', async () => {
  const releaseKeep = keepRoute('homelab', 'default')

  await openGatewayForAgent('homelab', 'default')
  const releaseRelay = retainGatewayForRelay('homelab', 'default')
  endpoint = 'wss://homelab.invalid/api/ws?token=new'
  disposeSecondariesForConnection('homelab', { redial: true })
  expect(gatewayMocks.connect).not.toHaveBeenCalledWith(endpoint)

  releaseRelay()
  await vi.waitFor(() => expect(gatewayMocks.connect).toHaveBeenLastCalledWith(endpoint))
  releaseKeep()
})

it('an unkept route is still pruned when idle and announces nothing', async () => {
  const opened = vi.fn()
  const off = onKeptRouteOpened(opened)

  await openGatewayForAgent('homelab', 'research')
  pastMinLifetime()
  pruneSecondaryGateways(new Set())
  await openGatewayForAgent('homelab', 'research')

  expect(gatewayMocks.connect).toHaveBeenCalledTimes(2)
  expect(opened).not.toHaveBeenCalled()
  off()
})
