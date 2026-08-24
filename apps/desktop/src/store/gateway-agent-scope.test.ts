import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// Registry-agent scoping regression (multi-source roster): the active key can
// name a `conn:<id>::<profile>` scope whose secondaries entry gets evicted by
// a teardown path (closeSecondaryGateways during a soft gateway switch,
// pruneSecondaryGateways). activeGateway() used to silently fall back to the
// PRIMARY gateway for a missing NAMED scope — every send/session op then hit
// the wrong machine with no error. These tests pin the invariant: a missing
// named scope never resolves to the primary; every eviction site explicitly
// re-points the active key at the primary so `activeKey` always resolves.

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
  $gateway,
  activeGateway,
  closeSecondaryGateways,
  configureGatewayRegistry,
  ensureGatewayForAgent,
  ensureGatewayForProfile,
  isActivePrimary,
  pruneSecondaryGateways,
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

describe('registry-agent scope eviction (activeGateway must never silently hit the primary)', () => {
  it('activates the agent socket, not the primary, for a registry scope', async () => {
    const primary = makePrimary()
    setPrimaryGateway(primary as never, 'default')
    installAgentDesktop()

    await ensureGatewayForAgent('homelab', 'research')

    expect(isActivePrimary()).toBe(false)
    expect(activeGateway()).not.toBe(primary)
    expect($gateway.get()).not.toBe(primary)
    expect(gatewayMocks.connect).toHaveBeenCalledWith(agentConn.wsUrl)
  })

  it('keeps the primary active when a fresh registry activation cannot resolve', async () => {
    const primary = makePrimary()
    setPrimaryGateway(primary as never, 'default')
    await ensureGatewayForProfile('default')
    const publishedPrimary = $gateway.get()
    installDesktop({
      getConnection: vi.fn(async () => agentConn),
      getConnectionFor: vi.fn(async () => {
        throw new Error('source unreachable')
      })
    })

    await expect(ensureGatewayForAgent('offline', 'research')).resolves.toBe(false)

    expect(isActivePrimary()).toBe(true)
    expect(activeGateway()).toBe(primary)
    expect($gateway.get()).toBe(publishedPrimary)
  })

  it('closeSecondaryGateways re-points the active key at the primary instead of dangling', async () => {
    const primary = makePrimary()
    setPrimaryGateway(primary as never, 'default')
    installAgentDesktop()

    await ensureGatewayForAgent('homelab', 'research')
    expect(isActivePrimary()).toBe(false)

    // The soft gateway switch path (use-gateway-boot) tears every secondary
    // down without knowing which one was active.
    closeSecondaryGateways()

    // Regression: activeKey used to keep naming the evicted scope while
    // activeGateway() fell back to the primary — wrong machine, no error.
    // Now the teardown restores the primary EXPLICITLY (atoms follow).
    expect(isActivePrimary()).toBe(true)
    expect(activeGateway()).toBe(primary)
    expect($gateway.get()).toBe(primary)
  })

  it('pruneSecondaryGateways never evicts the active scope and keeps the invariant', async () => {
    const primary = makePrimary()
    setPrimaryGateway(primary as never, 'default')
    installAgentDesktop()

    await ensureGatewayForAgent('homelab', 'research')
    const agentGateway = activeGateway()

    // Empty keep-set: everything non-active is evictable — the active agent
    // scope must survive, and the active pointer must still resolve to it.
    pruneSecondaryGateways(new Set())

    expect(isActivePrimary()).toBe(false)
    expect(activeGateway()).toBe(agentGateway)
    expect($gateway.get()).toBe(agentGateway)
  })
})
