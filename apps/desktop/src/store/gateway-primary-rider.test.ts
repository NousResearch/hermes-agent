import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// gatewayRouteState keys a profile that rode the shared primary socket as
// `primary:<profile>`. That hint belongs to the primary socket it was learned
// on: after a soft re-home (connection apply) the same profile may be served by
// a different backend, and keying it under the new primary would merge two
// owners into one roster source and skip a backend.

vi.mock('@/hermes', () => ({
  setApiRequestConnection: vi.fn(),
  HermesGateway: class {
    connectionState = 'closed'
    connect = vi.fn()
    close = vi.fn()
    onEvent = vi.fn(() => () => {})
    onState = vi.fn(() => () => {})
  }
}))
vi.mock('@/store/session', () => ({ setConnection: vi.fn(), setGatewayState: vi.fn() }))
vi.mock('@/store/notify-baseline', () => ({ markNativeNotifyBaseline: vi.fn() }))

const {
  closeSecondaryGateways,
  configureGatewayRegistry,
  gatewayRouteState,
  requestGatewayForProfile,
  setPrimaryGateway,
  setPrimaryGatewayConnectionId
} = await import('./gateway')

const shared = () => ({
  getConnection: vi.fn(async () => ({ port: 4242, profile: 'venture', sharedPrimary: true, token: 't' }))
})

const primary = () => ({ connectionState: 'open', request: vi.fn(async () => ({})) })

beforeEach(() => {
  configureGatewayRegistry({ onEvent: vi.fn(), primaryProfile: 'default' } as never)
  ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = shared()
})

afterEach(() => {
  closeSecondaryGateways()
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
})

describe('primary rider hint', () => {
  it('keys a shared profile on the primary it was learned on, and drops it when the primary socket changes', async () => {
    setPrimaryGateway(primary() as never, 'default')
    await requestGatewayForProfile('venture', 'subagent.roster')

    expect(gatewayRouteState(null, 'venture').key).toBe('primary:venture')

    setPrimaryGateway(primary() as never, 'default')

    expect(gatewayRouteState(null, 'venture').key).toBe('venture')
  })

  it('drops the hint when the same primary socket is re-homed to another connection', async () => {
    setPrimaryGateway(primary() as never, 'default')
    setPrimaryGatewayConnectionId('conn-a', 'remote')
    await requestGatewayForProfile('venture', 'subagent.roster')

    expect(gatewayRouteState(null, 'venture').key).toBe('primary:venture')

    setPrimaryGatewayConnectionId('conn-b', 'remote')

    expect(gatewayRouteState(null, 'venture').key).toBe('venture')
  })
})
