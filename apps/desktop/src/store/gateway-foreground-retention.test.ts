import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// #93892 regression: the registry disposed the secondary socket an idle Bot
// Chat tile was bound to — the live-work pruner for a pre-dialed (retained)
// entry, and the refcount-0 request lease for one the tile's own resume
// created. openGatewayForAgent marks the entry `retained`, but `retained` is
// a one-way latch every hover pre-warm and profile switch also sets, so no
// dispose path can honor it without leaking every socket ever warmed.
// Instead the registry's `foregroundScopes` hook names the scopes FOREGROUND
// surfaces are bound to (foregroundSessionScopes): a mounted tile pins its
// owner socket for exactly as long as it is mounted, on every dispose path.

const gatewayMocks = vi.hoisted(() => ({
  closed: [] as string[]
}))

vi.mock('@/hermes', async importActual => ({
  ...(await importActual<Record<string, unknown>>()),
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
    request = async (): Promise<unknown> => ({})
    onEvent = vi.fn(() => () => {})
    onState = vi.fn(() => () => {})
  }
}))
vi.mock('@/store/notify-baseline', () => ({ markNativeNotifyBaseline: vi.fn() }))

const {
  closeSecondaryGateways,
  configureGatewayRegistry,
  openGatewayForAgent,
  pruneSecondaryGateways,
  requestGatewayForAgent,
  setPrimaryGateway
} = await import('./gateway')

const { $sessionTiles, foregroundSessionScopes, liveSessionScopes } = await import('./session-states')

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

// What use-gateway-boot's recomputeKeptGateways feeds the pruner for a
// window with NO busy / needs-input work anywhere — the idle bot chat case.
// Foreground pins are NOT in here: the registry reads them itself.
const idleKeepSet = () => liveSessionScopes()

const BOT_TILE = {
  ownerRoute: { connectionId: 'local', mode: 'local' as const, profile: 'bot' },
  runtimeId: 'rt-bot',
  storedSessionId: 'stored-bot'
}

beforeEach(() => {
  installDesktop()
  // Wired exactly as use-gateway-boot wires it.
  configureGatewayRegistry({ foregroundScopes: foregroundSessionScopes, onEvent: vi.fn() })
  setPrimaryGateway({ connectionState: 'open' } as never, 'default')
  gatewayMocks.closed = []
  $sessionTiles.set([])
})

afterEach(() => {
  closeSecondaryGateways()
  $sessionTiles.set([])
  vi.clearAllMocks()
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
})

describe('foreground tile retention vs. the live-work pruner (#93892)', () => {
  it('keeps an idle Bot Chat tile’s owner socket across prune recomputes', async () => {
    // The BOTS workspace dials the bot's own backend without activating it
    // (keepAllProfilesScope) and opens the canonical chat as a tile on that
    // route. The global active profile stays primary/default.
    await openGatewayForAgent('local', 'bot')
    $sessionTiles.set([BOT_TILE])

    // Idle: no working / needs-input session anywhere. Before the fix this
    // recompute closed the socket → backend reaped the runtime → reclaim →
    // unbind → resume → … forever.
    pruneSecondaryGateways(idleKeepSet())
    pruneSecondaryGateways(idleKeepSet())

    expect(gatewayMocks.closed).toEqual([])
  })

  it('keeps the socket while the tile is still resuming (no runtime bound yet)', async () => {
    await openGatewayForAgent('local', 'bot')
    $sessionTiles.set([{ ...BOT_TILE, runtimeId: undefined }])

    pruneSecondaryGateways(idleKeepSet())

    expect(gatewayMocks.closed).toEqual([])
  })

  it('releases the socket once the tile is closed — the pin never latches', async () => {
    await openGatewayForAgent('local', 'bot')
    $sessionTiles.set([BOT_TILE])
    pruneSecondaryGateways(idleKeepSet())
    expect(gatewayMocks.closed).toEqual([])

    $sessionTiles.set([])
    pruneSecondaryGateways(idleKeepSet())

    expect(gatewayMocks.closed).toEqual(['wss://local.invalid/api/ws?profile=bot'])
  })

  it('keeps the socket a tile’s OWN resume dialed (no pre-dial, refcount-0 lease path)', async () => {
    // Relaunch with a persisted Bot Chat tab: nothing pre-dials the bot, so
    // the tile's session.resume goes out through requestGatewayForAgent's
    // per-request lease on a fresh, NON-retained entry. Before the fix the
    // lease's `finally` disposed that socket the instant the resume returned
    // — the runtime it had just minted was orphaned on the spot.
    $sessionTiles.set([{ ...BOT_TILE, runtimeId: undefined }])

    await requestGatewayForAgent('local', 'bot', 'session.resume', { session_id: 'stored-bot' })

    expect(gatewayMocks.closed).toEqual([])

    // …and the pruner agrees with the lease.
    pruneSecondaryGateways(idleKeepSet())
    expect(gatewayMocks.closed).toEqual([])

    // Tile gone: the next lease release disposes as it always did.
    $sessionTiles.set([])
    await requestGatewayForAgent('local', 'bot', 'session.usage', { session_id: 'stored-bot' })
    expect(gatewayMocks.closed).toEqual(['wss://local.invalid/api/ws?profile=bot'])
  })

  it('still prunes a merely pre-warmed (retained, no surface) socket — `retained` is not a pin', async () => {
    // A roster hover warms the bot's socket through the same door and sets
    // the same `retained` flag; with no tile bound to it, it is idle garbage.
    await openGatewayForAgent('local', 'bot')

    pruneSecondaryGateways(idleKeepSet())

    expect(gatewayMocks.closed).toEqual(['wss://local.invalid/api/ws?profile=bot'])
  })

  it('does not let a tile on one source pin another source’s same-named profile', async () => {
    await openGatewayForAgent('homelab', 'bot')
    $sessionTiles.set([BOT_TILE])

    pruneSecondaryGateways(idleKeepSet())

    expect(gatewayMocks.closed).toEqual(['wss://homelab.invalid/api/ws?profile=bot'])
  })
})
