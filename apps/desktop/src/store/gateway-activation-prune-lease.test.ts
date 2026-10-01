import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { RECONNECT_ATTEMPT_TIMEOUT_MS } from '@/lib/with-timeout'

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
  acquireGatewayRequestLease,
  acquireGatewayRequestLeaseForAgent,
  closeSecondaryGateways,
  configureGatewayRegistry,
  ensureGatewayForAgent,
  ensureGatewayForProfile,
  pruneSecondaryGateways,
  setPrimaryGateway,
  SECONDARY_MIN_LIFETIME_MS
} = await import('./gateway')

const { SOURCE_SWITCH_DIAL_TIMEOUT_MS } = await import('@/lib/with-timeout')

function installDesktop(): void {
  ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = {
    getConnection: vi.fn(async (profile: null | string) =>
      profile
        ? { port: 5151, profile, token: 'secondary-token', wsUrl: 'ws://127.0.0.1:5151/ws' }
        : { port: 4242, token: 'primary-token', wsUrl: 'ws://127.0.0.1:4242/ws' }
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
    connect: vi.fn(async function (this: { connectionState: string }) {
      this.connectionState = 'open'
    }),
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

// Flush microtasks until the dial registers its secondary (or bail). The dial
// path's hop count is an implementation detail — #95343's withTimeout wrapper
// added an await hop and broke the previous exactly-two-Promise.resolve flush.
async function flushUntilSecondaryRegistered(max = 20): Promise<void> {
  for (let i = 0; i < max && secondaryGateways.length === 0; i++) {
    await Promise.resolve()
  }
}

describe('activation lease vs. the live-work pruner (#89622)', () => {
  it('leases the exact connection when two sources expose the same profile', async () => {
    await ensureGatewayForAgent('source-a', 'default')
    await ensureGatewayForAgent('source-b', 'default')
    expect(secondaryGateways).toHaveLength(2)

    const lease = acquireGatewayRequestLeaseForAgent('source-b', 'default')

    expect(lease.ownerRoute).toEqual({ connectionId: 'source-b', profile: 'default' })
    await expect(lease.request('session.branch', { session_id: 'runtime-b' })).resolves.toEqual({
      method: 'session.branch',
      params: { session_id: 'runtime-b' }
    })
    expect(secondaryGateways[0].request).not.toHaveBeenCalled()
    expect(secondaryGateways[1].request).toHaveBeenCalledOnce()
    lease.release()
  })

  it('recovers a leased owner only when the socket is closed before dispatch', async () => {
    await ensureGatewayForProfile('bot')
    const gateway = secondaryGateways[0]
    const lease = acquireGatewayRequestLease(gateway as never, 'bot')
    gateway.connectionState = 'closed'

    await expect(lease.request('session.branch', { session_id: 'runtime-bot' })).resolves.toEqual({
      method: 'session.branch',
      params: { session_id: 'runtime-bot' }
    })
    expect(gateway.connect).toHaveBeenCalledTimes(2)
    expect(gateway.request).toHaveBeenCalledOnce()
    lease.release()
  })

  it('does not replay a leased mutation after a post-send connection close', async () => {
    await ensureGatewayForProfile('bot')
    const gateway = secondaryGateways[0]
    const lease = acquireGatewayRequestLease(gateway as never, 'bot')
    const outcomeUnknown = new Error('connection closed')

    gateway.request.mockImplementationOnce(async () => {
      gateway.connectionState = 'closed'
      throw outcomeUnknown
    })

    await expect(lease.request('session.branch', { session_id: 'runtime-bot' })).rejects.toBe(outcomeUnknown)
    expect(gateway.request).toHaveBeenCalledOnce()
    expect(gateway.connect).toHaveBeenCalledOnce()
    lease.release()
  })

  it('bounds a wedged primary descriptor lookup and permits a later recovery attempt', async () => {
    vi.useFakeTimers()
    const primary = makePrimary()
    const desktop = window.hermesDesktop
    vi.mocked(desktop.getConnection).mockImplementationOnce(() => new Promise(() => undefined))
    primary.connectionState = 'closed'
    setPrimaryGateway(primary as never, 'default')
    const firstLease = acquireGatewayRequestLease(primary as never, 'default')
    const firstRequest = firstLease.request('session.branch', { session_id: 'runtime-primary' })
    const firstRejection = expect(firstRequest).rejects.toThrow('Timed out reconnecting to Hermes backend')

    await vi.advanceTimersByTimeAsync(RECONNECT_ATTEMPT_TIMEOUT_MS)
    await firstRejection
    firstLease.release()

    const secondLease = acquireGatewayRequestLease(primary as never, 'default')
    await expect(secondLease.request('session.branch', { session_id: 'runtime-primary' })).resolves.toEqual({
      method: 'session.branch',
      params: { session_id: 'runtime-primary' }
    })
    expect(desktop.getConnection).toHaveBeenCalledTimes(2)
    secondLease.release()
  })

  it('bounds a wedged primary WebSocket URL refresh before dispatch', async () => {
    vi.useFakeTimers()
    const primary = makePrimary()
    const desktop = window.hermesDesktop
    vi.mocked(desktop.getConnection).mockResolvedValue({
      authMode: 'oauth',
      profile: 'default',
      wsUrl: 'wss://gateway.invalid/ws'
    } as never)
    desktop.getGatewayWsUrl = vi.fn(() => new Promise<string>(() => undefined))
    primary.connectionState = 'closed'
    setPrimaryGateway(primary as never, 'default')
    const lease = acquireGatewayRequestLease(primary as never, 'default')
    const request = lease.request('session.branch', { session_id: 'runtime-primary' })
    const rejection = expect(request).rejects.toThrow('Timed out re-minting the gateway WebSocket URL')

    await vi.advanceTimersByTimeAsync(RECONNECT_ATTEMPT_TIMEOUT_MS)
    await rejection
    expect(primary.connect).not.toHaveBeenCalled()
    expect(primary.request).not.toHaveBeenCalled()
    lease.release()
  })

  it('a prune during the switch dial does not dispose the target and the switch lands', async () => {
    let releaseConnect: () => void = () => undefined
    connectGate = new Promise<void>(resolve => {
      releaseConnect = resolve
    })

    const switching = ensureGatewayForProfile('bot')
    // Let the dial start (createSecondary + connect() now pending on the gate).
    await flushUntilSecondaryRegistered()
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
    await flushUntilSecondaryRegistered()
    expect(secondaryGateways).toHaveLength(1)

    pruneSecondaryGateways(new Set())
    expect(secondaryGateways[0].close).not.toHaveBeenCalled()

    releaseConnect()
    await expect(switching).resolves.toBe(true)
  })

  it('releasing a command lease cannot consume a prune deferred by an in-flight activation', async () => {
    let releaseConnect: () => void = () => undefined
    connectGate = new Promise<void>(resolve => {
      releaseConnect = resolve
    })

    const switching = ensureGatewayForProfile('bot')
    await flushUntilSecondaryRegistered()
    expect(secondaryGateways).toHaveLength(1)

    const commandLease = acquireGatewayRequestLease(secondaryGateways[0] as never, 'bot')

    pruneSecondaryGateways(new Set())
    commandLease.release()

    expect(secondaryGateways[0].close).not.toHaveBeenCalled()

    releaseConnect()
    await switching

    expect(secondaryGateways[0].connectionState).toBe('open')
  })

  it('the lease is released once the switch settles — a later prune reclaims the idle entry', async () => {
    await ensureGatewayForProfile('bot')

    // Move away so 'bot' is neither active nor kept, then prune: with the
    // lease released, the idle entry must be disposed exactly as before.
    await ensureGatewayForProfile('default')

    // Age the socket past the min-lifetime grace so this prune asserts the
    // lease release, not the freshly-opened spare (#94769).
    vi.useFakeTimers({ now: Date.now() + SECONDARY_MIN_LIFETIME_MS + 1_000 })
    pruneSecondaryGateways(new Set())
    vi.useRealTimers()

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
    await flushUntilSecondaryRegistered()
    expect(secondaryGateways).toHaveLength(1)

    // Inside the lease window: spared.
    pruneSecondaryGateways(new Set())
    expect(secondaryGateways[0].close).not.toHaveBeenCalled()

    // Past the lease window — which now spans a whole dial budget (#120298) —
    // reclaimed. (Lease is wall-clock bounded so a leaked lease cannot pin a
    // dead entry forever.)
    vi.setSystemTime(Date.now() + SOURCE_SWITCH_DIAL_TIMEOUT_MS + SECONDARY_MIN_LIFETIME_MS)
    pruneSecondaryGateways(new Set())
    expect(secondaryGateways[0].close).toHaveBeenCalled()

    releaseConnect()
  })

  it('a freshly opened idle secondary rides one prune tick before reaping (#94769)', async () => {
    vi.useFakeTimers()

    // A socket that just opened is a prune ↔ on-demand-dial race in the making:
    // its consumer may not have registered in the keep-set yet, and closing it
    // detaches the runtime → backend orphan-reap → `session.reclaimed` →
    // re-resume on a fresh socket the next recompute closes again. The
    // min-lifetime grace bounds that race without pinning the entry forever.
    await ensureGatewayForProfile('bot')
    expect(secondaryGateways[0].connectionState).toBe('open')

    // Move away so 'bot' is neither active nor in the keep-set.
    await ensureGatewayForProfile('default')
    expect(secondaryGateways).toHaveLength(1)

    // Immediately after open: spared by the grace window.
    pruneSecondaryGateways(new Set())
    expect(secondaryGateways[0].close).not.toHaveBeenCalled()

    // Past the grace window: reclaimed as idle, as before.
    vi.setSystemTime(Date.now() + SECONDARY_MIN_LIFETIME_MS + 1_000)
    pruneSecondaryGateways(new Set())
    expect(secondaryGateways[0].close).toHaveBeenCalled()
  })

  it('a dial the switch budget still allows is spared for its whole window, not just the first 30 s (#120298)', async () => {
    vi.useFakeTimers()

    try {
      let releaseConnect: () => void = () => undefined
      connectGate = new Promise<void>(resolve => {
        releaseConnect = resolve
      })

      const switching = ensureGatewayForProfile('bot')
      await flushUntilSecondaryRegistered()
      expect(secondaryGateways).toHaveLength(1)

      // The dial budget covers a whole remote bring-up chain (ssh connect, the
      // platform/locate/version probes, the remote spawn's ready sentinel, the
      // forward). A dial late in that window is still a dial the switch is
      // waiting on, so a recompute landing here must find the target leased —
      // otherwise the pruner disposes it, the bring-up lands afterwards, and
      // the switch resolves on a socket that openSecondary has already closed.
      vi.setSystemTime(Date.now() + SOURCE_SWITCH_DIAL_TIMEOUT_MS - 1_000)
      pruneSecondaryGateways(new Set())
      expect(secondaryGateways[0].close).not.toHaveBeenCalled()

      releaseConnect()
      await switching

      expect(secondaryGateways[0].connectionState).toBe('open')
    } finally {
      vi.useRealTimers()
    }
  })
})
