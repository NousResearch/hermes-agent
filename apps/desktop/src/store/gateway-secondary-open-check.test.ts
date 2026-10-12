import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// Regression for issue #92265: a transient first-dial WebSocket failure
// (e.g. ECONNRESET before the socket reaches `open`) must not let Desktop
// publish the closed gateway as the active route. entry.connection is set
// BEFORE the dial completes in openSecondary(), so checking only its
// truthiness previously let a failed activation still "succeed" and
// publish -- the next chat RPC then failed with "Hermes gateway is not
// connected" even though the UI had already switched to that route.

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
      // Unlike gateway-agent-scope.test.ts's always-succeeds mock, this
      // one propagates gatewayMocks.connect's outcome -- letting tests
      // below simulate a rejected first dial without flipping
      // connectionState to 'open'.
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
  $gateway,
  activeGateway,
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
}

function installDesktop(stub: DesktopStub): void {
  ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = stub
}

function makePrimary(): { connectionState: string } {
  return { connectionState: 'open' }
}

const agentConn = {
  authMode: 'token',
  baseUrl: 'https://homelab.invalid',
  mode: 'remote',
  profile: 'research',
  token: 'fake-test-token',
  wsUrl: 'wss://homelab.invalid/api/ws?token=fake-test-token'
}

function installAgentDesktop(): DesktopStub {
  const stub: DesktopStub = {
    getConnection: vi.fn(async () => agentConn),
    getConnectionFor: vi.fn(async () => agentConn)
  }

  installDesktop(stub)

  return stub
}

beforeEach(() => {
  configureGatewayRegistry({ onEvent: vi.fn() })
})

afterEach(() => {
  closeSecondaryGateways()
  vi.clearAllMocks()
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
})

describe('secondary activation requires an open socket, not just a connection descriptor (issue #92265)', () => {
  it('ensureGatewayForAgent: a transient first-dial failure does not activate or publish the closed gateway', async () => {
    const primary = makePrimary()
    setPrimaryGateway(primary as never, 'default')
    await ensureGatewayForProfile('default')
    const publishedPrimary = $gateway.get()
    installAgentDesktop()

    gatewayMocks.connect.mockRejectedValueOnce(new Error('ECONNRESET'))

    const activated = await ensureGatewayForAgent('homelab', 'research')

    expect(activated).toBe(false)
    // The exact reported symptom: the UI must NOT have switched away from
    // the primary onto the closed secondary.
    expect(isActivePrimary()).toBe(true)
    expect(activeGateway()).toBe(primary)
    expect($gateway.get()).toBe(publishedPrimary)
  })

  it('ensureGatewayForAgent: a successful dial still activates and publishes normally', async () => {
    const primary = makePrimary()
    setPrimaryGateway(primary as never, 'default')
    installAgentDesktop()

    const activated = await ensureGatewayForAgent('homelab', 'research')

    expect(activated).toBe(true)
    expect(isActivePrimary()).toBe(false)
    expect(activeGateway()).not.toBe(primary)
    expect($gateway.get()).not.toBe(primary)
  })

  it('ensureGatewayForProfile: a transient first-dial failure does not publish the closed gateway', async () => {
    const primary = makePrimary()
    setPrimaryGateway(primary as never, 'default')
    await ensureGatewayForProfile('default')
    const publishedPrimary = $gateway.get()
    installDesktop({
      getConnection: vi.fn(async () => agentConn),
      getConnectionFor: vi.fn(async () => agentConn)
    })

    gatewayMocks.connect.mockRejectedValueOnce(new Error('ECONNRESET'))

    // Post-#81165 the profile door RE-THROWS on a failed dial (so callers can
    // surface it); the #92265 invariant under test is unchanged — the closed
    // secondary must never be published as the active route.
    await expect(ensureGatewayForProfile('research')).rejects.toThrow('ECONNRESET')

    // Must still be on the primary -- the closed secondary was never
    // published as the active route.
    expect(isActivePrimary()).toBe(true)
    expect($gateway.get()).toBe(publishedPrimary)
  })

  it('ensureGatewayForProfile: a successful dial still activates and publishes normally', async () => {
    const primary = makePrimary()
    setPrimaryGateway(primary as never, 'default')
    installAgentDesktop()

    await ensureGatewayForProfile('research')

    expect(isActivePrimary()).toBe(false)
    expect($gateway.get()).not.toBe(primary)
  })
})
