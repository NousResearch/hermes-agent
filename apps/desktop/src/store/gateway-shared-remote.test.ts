import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// The global-remote share (backend routing case 3): every profile is served
// by the PRIMARY backend over one host, and getConnection() explicitly tags
// the shared descriptor with `sharedPrimary`. Dialing a second WebSocket at it
// used to fail over SSH (per-backend tunnel/ticket) and poison the active
// gateway with a closed socket — "Hermes gateway is not connected" for every
// profile except the primary. Pooled backends (own-remote override, local
// named profile) also carry `profile` for WS URL minting, so `profile` alone
// cannot identify the shared-primary route. These tests pin the fix: only a
// `sharedPrimary` descriptor activates the primary socket; a pooled descriptor
// that also carries `profile` must still dial its own socket.

const gatewayMocks = vi.hoisted(() => {
  const instances: Array<{ close: ReturnType<typeof vi.fn>; connectionState: string }> = []

  return {
    connect: vi.fn(async (_wsUrl: string): Promise<void> => {
      throw new Error('dialed a socket for a shared-primary profile')
    }),
    instances,
    setConnection: vi.fn()
  }
})

vi.mock('@/hermes', () => ({
  setApiRequestConnection: vi.fn(),
  HermesGateway: class {
    connectionState = 'closed'
    connect = async (wsUrl: string): Promise<void> => {
      await gatewayMocks.connect(wsUrl)
      this.connectionState = 'open'
    }
    close = vi.fn()
    onEvent = vi.fn(() => () => {})
    onState = vi.fn(() => () => {})

    constructor() {
      gatewayMocks.instances.push(this as never)
    }
  }
}))
vi.mock('@/store/session', () => ({
  setConnection: gatewayMocks.setConnection,
  setGatewayState: vi.fn()
}))
vi.mock('@/store/notify-baseline', () => ({ markNativeNotifyBaseline: vi.fn() }))

const {
  $gateway,
  activeGateway,
  closeSecondaryGateways,
  configureGatewayRegistry,
  ensureActiveGatewayOpen,
  ensureGatewayForProfile,
  setPrimaryGateway
} = await import('./gateway')

type DesktopStub = { getConnection: ReturnType<typeof vi.fn> }

function installDesktop(stub: DesktopStub): void {
  ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = stub
}

function makePrimary(): { connectionState: string } {
  // Only connectionState is consulted by setActive/isOpen for these paths.
  return { connectionState: 'open' }
}

const workerConnection = {
  authMode: 'token',
  baseUrl: 'https://worker.invalid',
  mode: 'remote',
  profile: 'worker',
  token: 'fake-test-token',
  wsUrl: 'wss://worker.invalid/api/ws?token=fake-test-token'
}

beforeEach(() => {
  configureGatewayRegistry({
    onEvent: vi.fn(),
    primaryProfile: 'default'
  } as never)
})

afterEach(() => {
  closeSecondaryGateways()
  gatewayMocks.instances.length = 0
  vi.clearAllMocks()
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
})

describe('ensureGatewayForProfile under a shared global remote', () => {
  it('activates the primary socket for an explicitly shared-primary descriptor', async () => {
    const primary = makePrimary()
    setPrimaryGateway(primary as never, 'default')
    installDesktop({
      // Shared descriptor: primary connection tagged with the profile scope
      // AND the explicit sharedPrimary marker.
      getConnection: vi.fn(async () => ({ port: 4242, profile: 'venture', sharedPrimary: true, token: 't' }))
    })

    await ensureGatewayForProfile('venture')

    expect(gatewayMocks.connect).not.toHaveBeenCalled()
    expect($gateway.get()).toBe(primary)
  })

  it('dials the exact WebSocket URL for a pooled profile descriptor that carries profile', async () => {
    const primary = makePrimary()
    const remoteWsUrl = 'wss://remote.invalid/api/ws?token=fake-test-token'

    setPrimaryGateway(primary as never, 'default')
    installDesktop({
      // Pooled descriptor: carries `profile` for WS URL minting but is NOT
      // shared-primary (no marker) — it must dial its own socket, not reuse
      // the primary. This is the local named / own-remote profile case.
      getConnection: vi.fn(async () => ({
        authMode: 'token',
        baseUrl: 'https://remote.invalid',
        mode: 'remote',
        profile: 'worker',
        token: 'fake-test-token',
        wsUrl: remoteWsUrl
      }))
    })
    gatewayMocks.connect.mockResolvedValueOnce(undefined)

    await ensureGatewayForProfile('worker')

    expect(gatewayMocks.connect).toHaveBeenCalledOnce()
    expect(gatewayMocks.connect).toHaveBeenCalledWith(remoteWsUrl)
    expect($gateway.get()).not.toBe(primary)
  })

  it('a transient first dial does not publish a closed gateway; the retry that opens it does (#92265)', async () => {
    const getConnection = vi.fn(async () => workerConnection)
    const primary = makePrimary()

    setPrimaryGateway(primary as never, 'default')
    installDesktop({ getConnection })

    // One cold-start dial transient-fails. (Once-queues only: clearAllMocks
    // keeps persistent implementations, so a bare mockRejectedValue would
    // leak into the next test.)
    gatewayMocks.connect
      .mockRejectedValueOnce(new Error('temporarily offline'))
      .mockResolvedValueOnce(undefined)

    await ensureGatewayForProfile('worker')

    // The previous (primary) route must stay published — a closed pooled
    // socket must not move the foreground or publish its connection.
    expect(gatewayMocks.setConnection).not.toHaveBeenCalled()
    expect(activeGateway()).toBe(primary)

    // The follow-up activation — the user's next click, after the backoff
    // loop healed the socket in the background — dials again, lands, and
    // publishes the connection exactly once.
    await ensureGatewayForProfile('worker')

    expect(gatewayMocks.setConnection).toHaveBeenCalledOnce()
    expect(gatewayMocks.setConnection).toHaveBeenLastCalledWith(workerConnection)
    expect(activeGateway()).not.toBe(primary)
    expect(activeGateway()?.connectionState).toBe('open')
  })

  it('re-publishes the active connection when the next RPC heals an already-active pooled socket', async () => {
    const getConnection = vi.fn(async () => workerConnection)
    const primary = makePrimary()

    setPrimaryGateway(primary as never, 'default')
    installDesktop({ getConnection })
    gatewayMocks.connect.mockResolvedValue(undefined)

    await ensureGatewayForProfile('worker')
    expect(gatewayMocks.setConnection).toHaveBeenCalledOnce()

    // The socket drops AFTER a successful activation.
    const socket = gatewayMocks.instances[0] as unknown as { connectionState: string }
    socket.connectionState = 'closed'

    // The next RPC drive (ensureActiveGatewayOpen) re-dials the ALREADY-ACTIVE
    // entry and re-publishes its descriptor — the recovery path the
    // optimistic entry.connection assignment exists for (reconnectSecondary
    // never applies the route; only an active entry's openSecondary publish
    // refreshes the connection).
    const healed = await ensureActiveGatewayOpen()

    expect(healed).not.toBeNull()
    expect((healed as unknown as { connectionState: string }).connectionState).toBe('open')
    expect(gatewayMocks.setConnection).toHaveBeenCalledTimes(2)
    expect(gatewayMocks.setConnection).toHaveBeenLastCalledWith(workerConnection)
  })
})
