import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// Connection lifecycle for registry-scoped secondary gateways:
//
//  1. Removing a connection must dispose its secondaries — remote/cloud
//     sources have no local process whose death would drop the socket, so
//     without an explicit dispose the WebSocket stays open and streams ghost
//     events until page reload.
//  2. A materially edited connection re-dials so fresh sockets target the
//     NEW endpoint.
//  3. When the Electron main reports the connection no longer exists
//     (`No connection with id`), the reconnect loop fail-stops and evicts
//     the entry instead of retrying forever.

const gatewayMocks = vi.hoisted(() => {
  const instances: { close: ReturnType<typeof vi.fn>; connectionState: string }[] = []

  return {
    connect: vi.fn(async (_wsUrl: string): Promise<void> => undefined),
    instances
  }
})

vi.mock('@/hermes', () => ({
  setApiRequestConnection: vi.fn(),
  HermesGateway: class {
    connectionState = 'closed'
    close = vi.fn(() => {
      this.connectionState = 'closed'
    })
    connect = async (wsUrl: string): Promise<void> => {
      await gatewayMocks.connect(wsUrl)
      this.connectionState = 'open'
    }
    onEvent = vi.fn(() => () => {})
    onState = vi.fn(() => () => {})
    constructor() {
      gatewayMocks.instances.push(this as never)
    }
  }
}))
vi.mock('@/store/session', () => ({
  setConnection: vi.fn(),
  setGatewayState: vi.fn()
}))
vi.mock('@/store/notify-baseline', () => ({ markNativeNotifyBaseline: vi.fn() }))

const { activeGateway, closeSecondaryGateways, configureGatewayRegistry, ensureGatewayForProfile, setPrimaryGateway } =
  await import('./gateway')

function installDesktop(stub: Record<string, unknown>): void {
  ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = stub
}

beforeEach(() => {
  configureGatewayRegistry({ onEvent: vi.fn() } as never)
  setPrimaryGateway({ connectionState: 'open' } as never, 'default')
})

afterEach(() => {
  closeSecondaryGateways()
  gatewayMocks.instances.length = 0
  vi.clearAllMocks()
  vi.useRealTimers()
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
})

describe('ensureGatewayForProfile — secondary connect failure surfaces (#81094)', () => {
  it('rethrows the dial failure instead of activating a closed socket', async () => {
    const getConnection = vi.fn(async ({ profile }: { profile: string }) => ({
      authMode: 'token',
      baseUrl: `https://${profile}.invalid`,
      mode: 'local',
      profile,
      token: 'fake-test-token',
      wsUrl: `wss://${profile}.invalid/ws`
    }))

    installDesktop({ getConnection })

    // First activation succeeds so the entry exists.
    await ensureGatewayForProfile('work')

    const live = activeGateway()

    expect(live).toBeTruthy()

    // The socket then dies (backend restart): state flips to closed, so the
    // next activation must re-dial instead of reusing the dead socket.
    ;(live as unknown as { connectionState: string }).connectionState = 'closed'
    gatewayMocks.connect.mockRejectedValue(new Error('backend unreachable'))

    await expect(ensureGatewayForProfile('work')).rejects.toThrow('backend unreachable')

    // The failed switch must NOT fall through to setActive() with a closed
    // socket: the active gateway is still the previously-live one, never the
    // dead entry that just failed to dial.
    const stillActive = activeGateway()

    expect(stillActive).toBe(live)
    expect(gatewayMocks.instances).toHaveLength(1)
  })

  it('keeps the reconnect schedule armed so transient failures still self-heal', async () => {
    vi.useFakeTimers()

    let failFirst = true

    const getConnection = vi.fn(async ({ profile }: { profile: string }) => ({
      authMode: 'token',
      baseUrl: `https://${profile}.invalid`,
      mode: 'local',
      profile,
      token: 'fake-test-token',
      wsUrl: `wss://${profile}.invalid/ws`
    }))

    installDesktop({ getConnection })

    gatewayMocks.connect.mockImplementation(async () => {
      if (failFirst) {
        throw new Error('backend unreachable')
      }
    })

    await expect(ensureGatewayForProfile('work')).rejects.toThrow('backend unreachable')

    // The catch kept the reconnect schedule: exactly one backoff timer is armed
    // for the failed entry (transient failures still self-heal).
    expect(vi.getTimerCount()).toBe(1)

    // Backoff fires → reconnect dials again → succeeds → socket opens.
    failFirst = false
    await vi.runAllTimersAsync()
    expect(gatewayMocks.instances[0].connectionState).toBe('open')
  })

  it('activates the secondary when connect succeeds', async () => {
    const getConnection = vi.fn(async ({ profile }: { profile: string }) => ({
      authMode: 'token',
      baseUrl: `https://${profile}.invalid`,
      mode: 'local',
      profile,
      token: 'fake-test-token',
      wsUrl: `wss://${profile}.invalid/ws`
    }))

    installDesktop({ getConnection })

    await ensureGatewayForProfile('work')

    expect(activeGateway()).toBe(gatewayMocks.instances[0])
  })
})
