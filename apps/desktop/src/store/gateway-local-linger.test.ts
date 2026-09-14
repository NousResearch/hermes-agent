import { registryBackendScopeKey } from '@hermes/shared'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// Local pooled-socket linger.
//
// #93594 (relay retention) and #93602 (session-sequence retention) both fixed
// dial-per-RPC churn, and both exempt the LOCAL source so the Electron idle
// reaper keeps its claim on spawned local backends. On a single-source desktop
// that exemption covers every route there is: each `conn:local::<profile>` scope
// re-dialed a fresh WebSocket per RPC, so a ~5s fan-out over N bots produced
// N connect/teardown pairs per tick, and any runtime session those RPCs minted
// died with its socket — the next call then returned 4001 "session not found".
//
// The linger keeps a local entry warm for a bounded window after its last
// request lease: long enough to coalesce a poll burst onto one socket, short
// enough to lapse between slow ticks so the backend stays idle-reap eligible.
// sweepIdleSecondaries() is the single expiry point.

const gatewayMocks = vi.hoisted(() => ({
  constructions: 0,
  closes: 0,
  connect: vi.fn(async (_wsUrl: string): Promise<void> => undefined),
  setConnection: vi.fn(),
  setGatewayState: vi.fn()
}))

vi.mock('@/hermes', () => ({
  setApiRequestConnection: vi.fn(),
  HermesGateway: class {
    connectionState = 'closed'
    constructor() {
      gatewayMocks.constructions += 1
    }
    connect = async (wsUrl: string): Promise<void> => {
      await gatewayMocks.connect(wsUrl)
      this.connectionState = 'open'
    }
    close = (): void => {
      gatewayMocks.closes += 1
      this.connectionState = 'closed'
    }
    request = async (): Promise<unknown> => ({})
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
  closeSecondaryGateways,
  configureGatewayRegistry,
  pruneSecondaryGateways,
  requestGatewayForAgent,
  retainGatewayForRelay,
  setPrimaryGateway,
  sweepIdleSecondaries
} = await import('./gateway')

const localConn = {
  authMode: 'none',
  baseUrl: 'http://127.0.0.1:0',
  mode: 'local',
  profile: 'chief-of-staff',
  wsUrl: 'ws://127.0.0.1:0/api/ws'
}

const remoteConn = {
  ...localConn,
  baseUrl: 'https://homelab.invalid',
  mode: 'remote',
  wsUrl: 'wss://homelab.invalid/api/ws'
}

function installDesktop(): void {
  ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = {
    getConnection: vi.fn(async () => localConn),
    getConnectionFor: vi.fn(async (connectionId: string) => (connectionId === 'local' ? localConn : remoteConn))
  }
}

beforeEach(() => {
  configureGatewayRegistry({ onEvent: vi.fn() })
  setPrimaryGateway({ connectionState: 'open' } as never, 'default')
  installDesktop()
  gatewayMocks.constructions = 0
  gatewayMocks.closes = 0
})

afterEach(() => {
  vi.useRealTimers()
  closeSecondaryGateways()
  vi.clearAllMocks()
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
})

describe('local pooled-socket linger', () => {
  it('coalesces a fan-out burst onto ONE socket instead of dialing per RPC', async () => {
    for (let tick = 0; tick < 5; tick += 1) {
      await requestGatewayForAgent('local', 'chief-of-staff', 'session.list', {})
    }

    // The churn this removes: 5 constructions, 5 teardowns. Now one socket
    // serves the whole burst.
    expect(gatewayMocks.constructions).toBe(1)
    expect(gatewayMocks.connect).toHaveBeenCalledTimes(1)
    expect(gatewayMocks.closes).toBe(0)
  })

  it('holds the socket across a session-scoped sequence so the runtime id survives', async () => {
    // The 4001 shape: session.create mints a runtime id on a socket that closed
    // before prompt.submit could use it. One socket across all three calls is
    // what keeps the gateway from reaping the session between them.
    await requestGatewayForAgent('local', 'chief-of-staff', 'session.create', {})
    await requestGatewayForAgent('local', 'chief-of-staff', 'session.attach', {})
    await requestGatewayForAgent('local', 'chief-of-staff', 'prompt.submit', {})

    expect(gatewayMocks.constructions).toBe(1)
    expect(gatewayMocks.closes).toBe(0)
  })

  it('remote routes keep dispose-at-refcount-0 — the linger is local-only', async () => {
    for (let tick = 0; tick < 3; tick += 1) {
      await requestGatewayForAgent('homelab', 'research', 'bot_relay.outbox.drain', {})
    }

    expect(gatewayMocks.constructions).toBe(3)
  })

  it('the live-work pruner does not evict a lingering entry between ticks', async () => {
    await requestGatewayForAgent('local', 'chief-of-staff', 'session.list', {})

    // A recompute landing mid-burst must not undo the linger, or the churn
    // returns through the pruner instead of the request lease.
    pruneSecondaryGateways(new Set())
    await requestGatewayForAgent('local', 'chief-of-staff', 'session.list', {})

    expect(gatewayMocks.constructions).toBe(1)
  })

  it('the sweep spares an entry whose linger is still open', async () => {
    await requestGatewayForAgent('local', 'chief-of-staff', 'session.list', {})

    sweepIdleSecondaries()
    await requestGatewayForAgent('local', 'chief-of-staff', 'session.list', {})

    expect(gatewayMocks.constructions).toBe(1)
    expect(gatewayMocks.closes).toBe(0)
  })

  it('the sweep disposes it once the linger lapses, restoring idle-reap eligibility', async () => {
    vi.useFakeTimers()

    await requestGatewayForAgent('local', 'chief-of-staff', 'session.list', {})
    expect(gatewayMocks.constructions).toBe(1)

    // LOCAL_IDLE_LINGER_MS is 20s; advance past it and sweep.
    vi.advanceTimersByTime(21_000)
    sweepIdleSecondaries()

    expect(gatewayMocks.closes).toBe(1)

    // The next RPC dials fresh, proving the entry was really evicted (and so
    // the backend spent that gap eligible for the reaper).
    await requestGatewayForAgent('local', 'chief-of-staff', 'session.list', {})
    expect(gatewayMocks.constructions).toBe(2)
  })

  it('the sweep spares an entry a foreground surface pinned AFTER the linger began (#93892)', async () => {
    vi.useFakeTimers()

    // The linger is stamped while nothing has this entry pinned: the lease
    // release only reaches that branch when foregroundPinned() is false.
    await requestGatewayForAgent('local', 'chief-of-staff', 'session.list', {})
    expect(gatewayMocks.constructions).toBe(1)

    // Only afterwards does a tile mount and bind a runtime to the same socket.
    // The pin therefore arrives too late for the linger to be what protects
    // it, which is why sweepIdleSecondaries() has to consult foregroundPinned
    // on its own. Upstream added that pin after this patch was written.
    configureGatewayRegistry({
      onEvent: vi.fn(),
      foregroundScopes: () => new Set([registryBackendScopeKey('local', 'chief-of-staff')])
    })

    vi.advanceTimersByTime(21_000)
    sweepIdleSecondaries()

    // A close here tears down a socket whose surface is still mounted, which
    // is the exact failure #93892 exists to prevent.
    expect(gatewayMocks.closes).toBe(0)
  })

  it('the sweep never evicts a socket with a request in flight', async () => {
    vi.useFakeTimers()

    await requestGatewayForAgent('local', 'chief-of-staff', 'session.list', {})
    vi.advanceTimersByTime(21_000)

    // Lapsed linger, but a concurrent call re-leased the entry: eviction here
    // would close a socket mid-RPC.
    const inFlight = requestGatewayForAgent('local', 'chief-of-staff', 'prompt.submit', {})
    // The request lease is one microtask out: upstream now awaits
    // isAttachedSharedRemote() before `entry.activeRequests += 1`, so a sweep
    // fired in that gap sees an unleased entry. Flush to the point the RPC
    // actually holds its lease, which is the state this guard is about.
    await Promise.resolve()
    await Promise.resolve()
    sweepIdleSecondaries()
    await inFlight

    expect(gatewayMocks.closes).toBe(0)
  })

  it('the sweep leaves remote entries alone even when idle', async () => {
    vi.useFakeTimers()

    const release = retainGatewayForRelay('homelab', 'research')
    await requestGatewayForAgent('homelab', 'research', 'bot_relay.outbox.drain', {})

    vi.advanceTimersByTime(21_000)
    sweepIdleSecondaries()

    // Relay retention is unbounded by design; the sweep must not undercut it.
    expect(gatewayMocks.closes).toBe(0)
    release()
  })
})
