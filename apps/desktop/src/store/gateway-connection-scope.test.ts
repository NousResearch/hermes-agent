import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// Cross-connection event/prune scoping: every registered source exposes a
// 'default' profile (the roster force-unshifts it), so any consumer keyed by
// the bare profile name conflates two connected gateways' activity. These
// tests pin the composite (connectionId, profile) keying:
//  - pruneSecondaryGateways must NOT keep a registry-scoped socket alive off
//    another source's same-named profile (and vice versa) — registry entries
//    match only their composite backendScopeKey scope.
//  - the session-states scope ledger (recordSessionEventScope /
//    liveSessionScopes) turns registry-tagged live work into those composite
//    keep-set entries, and ignores untagged local/primary events.

const gatewayMocks = vi.hoisted(() => ({
  closed: [] as string[],
  setConnection: vi.fn()
}))

vi.mock('@/hermes', () => ({
  setApiRequestConnection: vi.fn(),
  HermesGateway: class {
    connectionState = 'closed'
    wsUrl = ''
    connect = async (wsUrl: string): Promise<void> => {
      this.wsUrl = wsUrl
      this.connectionState = 'open'
    }
    close = (): void => {
      gatewayMocks.closed.push(this.wsUrl)
      this.connectionState = 'closed'
    }
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
  closeSecondaryGateways,
  configureGatewayRegistry,
  ensureGatewayForAgent,
  openGatewayForAgent,
  pruneSecondaryGateways,
  setPrimaryGateway
} = await import('./gateway')

function installDesktop(): void {
  ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = {
    getConnection: vi.fn(async () => ({
      authMode: 'token',
      profile: 'default',
      token: 't',
      wsUrl: 'wss://local.invalid/api/ws?token=t'
    })),
    getConnectionFor: vi.fn(async ({ connectionId, profile }: { connectionId: string; profile: string }) => ({
      authMode: 'token',
      connectionId,
      profile,
      token: 't',
      wsUrl: `wss://${connectionId}.invalid/api/ws?profile=${profile}`
    })),
    getGatewayWsUrlFor: vi.fn(
      async ({ connectionId, profile }: { connectionId: string; profile: string }) =>
        `wss://${connectionId}.invalid/api/ws?profile=${profile}`
    ),
    touchBackend: vi.fn(async () => undefined)
  }
}

beforeEach(() => {
  installDesktop()
  configureGatewayRegistry({ onEvent: vi.fn() })
  setPrimaryGateway({ connectionState: 'open' } as never, 'default')
  gatewayMocks.closed = []
})

afterEach(() => {
  closeSecondaryGateways()
  vi.clearAllMocks()
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
})

describe('pruneSecondaryGateways with registry-scoped entries', () => {
  it('keeps the previous source socket open when Sessions switches backends', async () => {
    await ensureGatewayForAgent('work', 'default')
    await ensureGatewayForAgent('homelab', 'default')

    // Source switching only changes the foreground route. Retaining the first
    // socket lets its live turn continue and keeps receiving completion events.
    expect(gatewayMocks.closed).toEqual([])
  })

  it("does not keep a registry socket alive off another source's same-named profile", async () => {
    // Gateway B's 'default' — the roster row (connectionId 'homelab', profile
    // 'default'). The keep-set carries the bare 'default' profile because the
    // LOCAL source has live work; that must not pin homelab's socket.
    await openGatewayForAgent('homelab', 'default')

    pruneSecondaryGateways(new Set(['default']))

    expect(gatewayMocks.closed).toEqual(['wss://homelab.invalid/api/ws?profile=default'])
  })

  it('a switch-phase dial (activationLease) survives a live-work recompute until its activation lands', async () => {
    // Phase one of the Sessions-switcher source switch: the target is opened
    // but not yet active and has no live work of its own. Another source's
    // streaming turn recomputes the keep-set mid-dial — that must not dispose
    // the socket the switch is about to activate (#89622 via #93937).
    await openGatewayForAgent('homelab', 'default', { activationLease: true })

    pruneSecondaryGateways(new Set(['default']))

    expect(gatewayMocks.closed).toEqual([])
  })

  it('keeps a registry socket whose composite scope has live work', async () => {
    await openGatewayForAgent('homelab', 'default')

    pruneSecondaryGateways(new Set(['conn:homelab::default']))

    expect(gatewayMocks.closed).toEqual([])
  })

  it('still keeps a local (profile-keyed) secondary via its bare profile name', async () => {
    await openGatewayForAgent(null, 'research')

    pruneSecondaryGateways(new Set(['research']))

    expect(gatewayMocks.closed).toEqual([])

    pruneSecondaryGateways(new Set())

    expect(gatewayMocks.closed).toHaveLength(1)
  })
})
