import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// #92265 regression: a secondary gateway activation could publish a CLOSED
// socket. openSecondary records entry.connection (optimistically — see the
// comment there) before the WebSocket dial, and the activation conditions
// accepted a non-null connection without requiring the gateway to be OPEN —
// a cold backend whose forwarded port answers TCP a beat before its WS
// endpoint is listening (transient ECONNRESET) became the foreground route
// anyway, and the next chat RPC died with "Hermes gateway is not connected".
// These tests pin:
//   1. a transient first dial refuses the activation — the previous route
//      stays published and nothing is published for the closed socket,
//   2. the follow-up activation publishes exactly once the socket is open,
//   3. a failed activation on a shared source cannot fall through to a
//      different profile's backend.

const gatewayMocks = vi.hoisted(() => ({
  connect: vi.fn(async (_wsUrl: string): Promise<void> => undefined),
  setConnection: vi.fn(),
  setGatewayState: vi.fn()
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
    onEvent = vi.fn(() => () => {})
    onState = vi.fn(() => () => {})
  }
}))
vi.mock('@/store/session', () => ({
  setConnection: gatewayMocks.setConnection,
  setGatewayState: gatewayMocks.setGatewayState
}))
vi.mock('@/store/notify-baseline', () => ({ markNativeNotifyBaseline: vi.fn() }))

const {
  $activeGatewayRoute,
  activeGateway,
  activeGatewayConnectionId,
  activeGatewayProfileKey,
  closeSecondaryGateways,
  configureGatewayRegistry,
  ensureGatewayForAgent,
  ensureGatewayForProfile,
  isActivePrimary,
  setPrimaryGateway
} = await import('./gateway')

interface DesktopStub {
  getConnection: ReturnType<typeof vi.fn>
  getConnectionFor: ReturnType<typeof vi.fn>
  touchBackend?: ReturnType<typeof vi.fn>
}

function installDesktop(stub: DesktopStub): void {
  ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = stub
}

function makePrimary(): { connectionState: string } {
  return { connectionState: 'open' }
}

function descriptorFor(connectionId: string, profile: string) {
  return {
    authMode: 'token',
    baseUrl: `https://${connectionId}.invalid`,
    mode: 'remote',
    profile,
    token: 'fake-test-token',
    wsUrl: `wss://${connectionId}.invalid/api/ws?profile=${profile}`
  }
}

function installAgentDesktop(): DesktopStub {
  const stub: DesktopStub = {
    getConnection: vi.fn(async () => descriptorFor('legacy-local', 'default')),
    getConnectionFor: vi.fn(async ({ connectionId, profile }: { connectionId: string; profile: string }) =>
      descriptorFor(connectionId, profile)
    ),
    touchBackend: vi.fn(async () => undefined)
  }

  installDesktop(stub)

  return stub
}

beforeEach(() => {
  configureGatewayRegistry({ onEvent: vi.fn() })
  setPrimaryGateway(makePrimary() as never, 'default')
})

afterEach(() => {
  closeSecondaryGateways()
  vi.clearAllMocks()
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
})

describe('secondary activation must never publish a closed gateway (#92265)', () => {
  it('a transient first dial keeps the previous registry route and publishes nothing', async () => {
    installAgentDesktop()
    gatewayMocks.connect.mockRejectedValueOnce(new Error('ECONNRESET'))

    await expect(ensureGatewayForAgent('homelab', 'research')).resolves.toBe(false)

    expect(gatewayMocks.connect).toHaveBeenCalledOnce()
    expect(isActivePrimary()).toBe(true)
    expect(activeGateway()?.connectionState).toBe('open')
    expect(gatewayMocks.setConnection).not.toHaveBeenCalled()
  })

  it('a failed profile dial keeps the previous route and publishes nothing', async () => {
    installAgentDesktop()
    gatewayMocks.connect.mockRejectedValueOnce(new Error('ECONNRESET'))

    await ensureGatewayForProfile('worker')

    // The profile path must not move the active route onto the closed socket.
    expect(gatewayMocks.connect).toHaveBeenCalledOnce()
    expect(isActivePrimary()).toBe(true)
    expect(activeGateway()?.connectionState).toBe('open')
    expect(gatewayMocks.setConnection).not.toHaveBeenCalled()
  })

  it('the follow-up activation publishes exactly once the socket is open', async () => {
    installAgentDesktop()
    gatewayMocks.connect.mockRejectedValueOnce(new Error('ECONNRESET'))

    // First click: the dial resets and the activation is refused.
    await expect(ensureGatewayForAgent('homelab', 'research')).resolves.toBe(false)

    // Follow-up click: the socket opens (the backoff loop may have healed it
    // in the meantime) and the route lands, publishing exactly once.
    await expect(ensureGatewayForAgent('homelab', 'research')).resolves.toBe(true)

    expect(gatewayMocks.connect).toHaveBeenCalledTimes(2)
    expect(isActivePrimary()).toBe(false)
    expect(activeGatewayProfileKey()).toBe('research')
    expect(activeGateway()?.connectionState).toBe('open')
    expect(gatewayMocks.setConnection).toHaveBeenCalledOnce()
  })

  it('a failed activation on a shared source cannot fall through to another profile', async () => {
    installAgentDesktop()

    // One open socket serves homelab's 'default' profile.
    await expect(ensureGatewayForAgent('homelab', 'default')).resolves.toBe(true)
    const opened = activeGateway()

    // The 'work' profile on the SAME source dial fails once.
    gatewayMocks.connect.mockRejectedValueOnce(new Error('ECONNRESET'))

    await expect(ensureGatewayForAgent('homelab', 'work')).resolves.toBe(false)

    // The route must still name homelab's OPEN 'default' socket — never a
    // closed 'work' socket that happens to share the same connection id.
    expect(activeGateway()).toBe(opened)
    expect(activeGatewayConnectionId()).toBe('homelab')
    expect(activeGatewayProfileKey()).toBe('default')
    expect($activeGatewayRoute.get()).toBe('default')
    expect((opened as unknown as { connectionState: string }).connectionState).toBe('open')
  })
})
