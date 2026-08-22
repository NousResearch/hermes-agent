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

const gatewayMocks = vi.hoisted(() => ({
  connect: vi.fn(async (_wsUrl: string): Promise<void> => {
    throw new Error('dialed a socket for a shared-primary profile')
  }),
  setConnection: vi.fn()
}))

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
  }
}))
vi.mock('@/store/session', () => ({
  setConnection: gatewayMocks.setConnection,
  setGatewayState: vi.fn()
}))
vi.mock('@/store/notify-baseline', () => ({ markNativeNotifyBaseline: vi.fn() }))

const {
  $gateway,
  closeSecondaryGateways,
  configureGatewayRegistry,
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

beforeEach(() => {
  configureGatewayRegistry({
    onEvent: vi.fn(),
    primaryProfile: 'default'
  } as never)
})

afterEach(() => {
  closeSecondaryGateways()
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

  it('rejects the failed dial without publishing an activation, then activates once the backend returns', async () => {
    const connection = {
      authMode: 'token',
      baseUrl: 'https://worker.invalid',
      mode: 'remote',
      profile: 'worker',
      token: 'fake-test-token',
      wsUrl: 'wss://worker.invalid/api/ws?token=fake-test-token'
    }

    const getConnection = vi.fn(async () => connection)

    setPrimaryGateway(makePrimary() as never, 'default')
    installDesktop({ getConnection })

    gatewayMocks.connect.mockRejectedValueOnce(new Error('temporarily offline')).mockResolvedValueOnce(undefined)

    // #81094: a failed dial must REJECT instead of silently activating a
    // closed socket that would route messages to the primary backend.
    await expect(ensureGatewayForProfile('worker')).rejects.toThrow('temporarily offline')

    // No activation was published for the dead dial — $connection keeps the
    // primary's descriptor (set by setPrimaryGateway), never the unreachable
    // secondary's. (Also the #92265 shape: publish requires an OPEN socket.)
    expect(gatewayMocks.setConnection).not.toHaveBeenCalled()

    // Once the backend is reachable again, retrying the switch activates and
    // publishes the live connection descriptor.
    await ensureGatewayForProfile('worker')

    expect(gatewayMocks.setConnection).toHaveBeenCalledTimes(1)
    expect(gatewayMocks.setConnection).toHaveBeenLastCalledWith(connection)
  })
})
