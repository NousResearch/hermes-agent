import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// Regression suite for #89622: clicking a profile in the rail did nothing.
// The live-work pruner (pruneSecondaryGateways) disposed the switch target's
// secondary entry while its socket was still dialing — the target is not yet
// the active key, has no live sessions, and holds no request lease, so every
// prune recompute during the ~3s cold backend spawn saw it as idle garbage.
// These tests pin the activation lease that spares a mid-dial entry, its
// release once the activation settles, and its bounded self-heal.

const secondaryGateways: Array<{
  close: ReturnType<typeof vi.fn>
  connect: ReturnType<typeof vi.fn>
  connectionState: string
  request: ReturnType<typeof vi.fn>
}> = []

let connectGate: Promise<void> | null = null

vi.mock('@/hermes', () => ({
  HermesGateway: class {
    connectionState = 'closed'
    connect = vi.fn(async () => {
      if (this.connectionState === 'connecting') {
        return
      }

      this.connectionState = 'connecting'

      if (connectGate) {
        await connectGate
      }

      this.connectionState = 'open'
    })
    request = vi.fn(async (method: string, params: Record<string, unknown>) => ({ method, params }))
    close = vi.fn()
    onEvent = vi.fn(() => () => {})
    onState = vi.fn(() => () => {})

    constructor() {
      secondaryGateways.push(this)
    }
  },
  setApiRequestConnection: vi.fn()
}))
vi.mock('@/store/session', () => ({ setConnection: vi.fn(), setGatewayState: vi.fn() }))
vi.mock('@/store/notify-baseline', () => ({ markNativeNotifyBaseline: vi.fn() }))

const {
  closeSecondaryGateways,
  configureGatewayRegistry,
  ensureGatewayForAgent,
  ensureGatewayForProfile,
  pruneSecondaryGateways,
  setPrimaryGateway
} = await import('./gateway')

function installDesktop(): void {
  ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = {
    getConnection: vi.fn(async (profile: null | string) =>
      profile ? { port: 5151, profile, token: 'secondary-token' } : { port: 4242, token: 'primary-token' }
    ),
    getConnectionFor: vi.fn(async ({ profile }: { connectionId: string; profile: string }) => ({
      port: 6161,
      profile,
      token: 'registry-token'
    })),
    getGatewayWsUrlFor: vi.fn(async () => 'ws://127.0.0.1:6161/ws'),
    touchBackend: vi.fn(async () => undefined)
  }
}

function makePrimary() {
  return {
    connectionState: 'open',
    request: vi.fn(async (method: string, params: Record<string, unknown>) => ({ method, params }))
  }
}

beforeEach(() => {
  secondaryGateways.length = 0
  connectGate = null
  configureGatewayRegistry({ onEvent: vi.fn() })
  closeSecondaryGateways()
  installDesktop()
  setPrimaryGateway(makePrimary() as never, 'default')
})

afterEach(() => {
  closeSecondaryGateways()
  vi.clearAllMocks()
  vi.useRealTimers()
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
})

describe('activation lease vs. the live-work pruner (#89622)', () => {
  it('a prune during the switch dial does not dispose the target and the switch lands', async () => {
    let releaseConnect: () => void = () => undefined
    connectGate = new Promise<void>(resolve => {
      releaseConnect = resolve
    })

    const switching = ensureGatewayForProfile('bot')
    // Let the dial start (createSecondary + connect() now pending on the gate).
    await Promise.resolve()
    await Promise.resolve()
    expect(secondaryGateways).toHaveLength(1)

    // The exact recompute that killed the switch: no live sessions, target not
    // active yet, empty keep-set. The mid-dial entry must survive it.
    pruneSecondaryGateways(new Set())
    expect(secondaryGateways[0].close).not.toHaveBeenCalled()

    releaseConnect()
    await switching

    // The switch landed on the target's own (still-live) socket.
    expect(secondaryGateways[0].connectionState).toBe('open')
  })

  it('the agent door leases its entry against the pruner too', async () => {
    let releaseConnect: () => void = () => undefined
    connectGate = new Promise<void>(resolve => {
      releaseConnect = resolve
    })

    const switching = ensureGatewayForAgent('homelab', 'research')
    await Promise.resolve()
    await Promise.resolve()
    expect(secondaryGateways).toHaveLength(1)

    pruneSecondaryGateways(new Set())
    expect(secondaryGateways[0].close).not.toHaveBeenCalled()

    releaseConnect()
    await expect(switching).resolves.toBe(true)
  })

  it('the lease is released once the switch settles — a later prune reclaims the idle entry', async () => {
    await ensureGatewayForProfile('bot')

    // Move away so 'bot' is neither active nor kept, then prune: with the
    // lease released, the idle entry must be disposed exactly as before.
    await ensureGatewayForProfile('default')

    pruneSecondaryGateways(new Set())

    expect(secondaryGateways[0].close).toHaveBeenCalled()
  })

  it('an orphaned lease expires — the pruner is not permanently locked out', async () => {
    vi.useFakeTimers()

    let releaseConnect: () => void = () => undefined
    connectGate = new Promise<void>(resolve => {
      releaseConnect = resolve
    })

    // Dial starts and never settles (wedged bridge call).
    void ensureGatewayForProfile('bot')
    await Promise.resolve()
    await Promise.resolve()
    expect(secondaryGateways).toHaveLength(1)

    // Inside the lease window: spared.
    pruneSecondaryGateways(new Set())
    expect(secondaryGateways[0].close).not.toHaveBeenCalled()

    // Past the lease window: reclaimed. (Lease is wall-clock bounded so a
    // leaked lease cannot pin a dead entry forever.)
    vi.setSystemTime(Date.now() + 31_000)
    pruneSecondaryGateways(new Set())
    expect(secondaryGateways[0].close).toHaveBeenCalled()

    releaseConnect()
  })
})
