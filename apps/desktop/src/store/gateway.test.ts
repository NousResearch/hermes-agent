import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// Module state lives in a module-local singleton (see gatewayState), so the
// mocks below must be in place before the gateway module is first imported.
const m = vi.hoisted(() => {
  class MockGateway {
    static instances: MockGateway[] = []
    // Per-test connect outcome; the instance is created inside
    // createSecondary, so behavior is registered on the class, not the node.
    static connectOutcome: 'resolve' | 'reject' = 'resolve'

    connectionState = 'closed'
    connect = vi.fn()
    close = vi.fn()
    onEvent = vi.fn(() => () => {})
    onState = vi.fn(() => () => {})

    constructor() {
      MockGateway.instances.push(this)
      this.connect.mockImplementation(async () => {
        if (MockGateway.connectOutcome === 'reject') {
          throw new Error('backend unreachable')
        }
      })
    }
  }

  return {
    MockGateway,
    hermesDesktop: {
      getConnection: vi.fn(),
      touchBackend: vi.fn()
    }
  }
})

vi.mock('@/hermes', () => ({ HermesGateway: m.MockGateway }))
vi.mock('@hermes/shared', () => ({
  resolveGatewayWsUrl: vi.fn(async () => 'ws://test')
}))
vi.mock('@/store/session', () => ({ setGatewayState: vi.fn() }))
vi.mock('@/store/notify-baseline', () => ({ markNativeNotifyBaseline: vi.fn() }))

import { $gateway, closeSecondaryGateways, ensureGatewayForProfile } from './gateway'

describe('ensureGatewayForProfile — secondary connect failure (#81094)', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    m.MockGateway.instances = []
    m.MockGateway.connectOutcome = 'resolve'
    m.hermesDesktop.getConnection.mockResolvedValue({ profile: 'secondary' })
    m.hermesDesktop.touchBackend.mockResolvedValue(undefined)
    // openSecondary bails out early when window.hermesDesktop is missing
    // (`if (!desktop) return`), which would silently resolve instead of
    // exercising the connect path under test.
    vi.stubGlobal('window', { hermesDesktop: m.hermesDesktop })
  })

  afterEach(() => {
    closeSecondaryGateways()
    vi.useRealTimers()
    vi.unstubAllGlobals()
  })

  it('rethrows the connect failure instead of activating a closed socket', async () => {
    m.MockGateway.connectOutcome = 'reject'

    await expect(ensureGatewayForProfile('secondary')).rejects.toThrow('backend unreachable')
    // The failure must NOT fall through to setActive with a closed socket.
    expect($gateway.get()).toBeNull()
  })

  it('keeps the reconnect schedule on failure so transient errors still self-heal', async () => {
    m.MockGateway.connectOutcome = 'reject'

    await expect(ensureGatewayForProfile('secondary')).rejects.toThrow('backend unreachable')
    // scheduleReconnect armed the backoff timer for the failed entry.
    expect(vi.getTimerCount()).toBe(1)
  })

  it('activates the secondary once connect succeeds', async () => {
    await ensureGatewayForProfile('secondary')

    // $gateway points at the entry created by createSecondary, not a
    // hand-made instance.
    expect($gateway.get()).toBe(m.MockGateway.instances[0])
  })
})
