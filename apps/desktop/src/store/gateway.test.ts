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

const {
  activeGateway,
  closeSecondaryGateways,
  configureGatewayRegistry,
  ensureGatewayForProfile,
  openGatewayForAgent,
  pruneSecondaryGateways,
  setPrimaryGateway
} = await import('./gateway')

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

  it('releases the activation lease when the first dial is rejected so pruning disposes it', async () => {
    const getConnection = vi.fn(async ({ profile }: { profile: string }) => ({
      authMode: 'token',
      baseUrl: `https://${profile}.invalid`,
      mode: 'local',
      profile,
      token: 'fake-test-token',
      wsUrl: `wss://${profile}.invalid/ws`
    }))

    installDesktop({ getConnection })
    gatewayMocks.connect.mockRejectedValue(new Error('backend unreachable'))

    await expect(ensureGatewayForProfile('work')).rejects.toThrow('backend unreachable')

    pruneSecondaryGateways(new Set())

    expect(gatewayMocks.instances[0].close).toHaveBeenCalledTimes(1)
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

describe('connection-scoped dial failure identity (#95421)', () => {
  it('logs the route scope while preserving the original dial error', async () => {
    const dialError = new Error('backend unreachable')

    const getConnectionFor = vi.fn(async ({ connectionId, profile }: { connectionId: string; profile: string }) => ({
      authMode: 'token',
      connectionId,
      profile,
      wsUrl: `wss://${connectionId}.invalid/ws`
    }))

    installDesktop({ getConnectionFor })
    gatewayMocks.connect.mockRejectedValue(dialError)

    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => undefined)

    try {
      await expect(openGatewayForAgent('work', 'default')).rejects.toBe(dialError)
      await expect(openGatewayForAgent('homelab', 'default')).rejects.toBe(dialError)

      const messages = errorSpy.mock.calls.map(([message]) => String(message))

      expect(messages).toHaveLength(2)
      expect(messages).toEqual(
        expect.arrayContaining([
          expect.stringContaining('scope="conn:work::default"'),
          expect.stringContaining('scope="conn:homelab::default"')
        ])
      )
      expect(messages.every(message => message.includes('profile="default"'))).toBe(true)
      expect(new Set(messages).size).toBe(2)
      expect(messages.join(' ')).not.toContain('wss://')

      for (const [, error] of errorSpy.mock.calls) {
        expect(error).toBe(dialError)
      }
    } finally {
      errorSpy.mockRestore()
    }
  })
})

describe('profile switch mid-WS-handshake (#92434 close-candidate pin)', () => {
  // Reported shape: Bot ↔ Default switching killed the socket until an app
  // restart. The activation-epoch guard (applyActive) + open-socket-publish
  // rule mean a switch-back that lands while the outgoing switch's handshake
  // is still pending must win the route, and the late-completing dial must
  // neither steal the foreground nor leave its socket permanently broken.
  it('a switch-back during a pending handshake wins; the late dial neither steals the route nor breaks the socket', async () => {
    const getConnection = vi.fn(async ({ profile }: { profile: string }) => ({
      authMode: 'token',
      baseUrl: `https://${profile}.invalid`,
      mode: 'local',
      profile,
      token: 'fake-test-token',
      wsUrl: `wss://${profile}.invalid/ws`
    }))

    installDesktop({ getConnection })

    let releaseDial: () => void = () => undefined

    gatewayMocks.connect.mockImplementation(
      () =>
        new Promise<void>(resolve => {
          releaseDial = resolve
        })
    )

    // 1. Default → Bot: the secondary's WS handshake starts and stays pending.
    const botActivation = ensureGatewayForProfile('bot')

    await vi.waitFor(() => expect(gatewayMocks.connect).toHaveBeenCalledTimes(1))

    // 2. The user switches back to Default while that handshake is mid-flight.
    await ensureGatewayForProfile('default')

    const primary = activeGateway()

    expect(primary).toBeTruthy()

    // 3. The Bot handshake completes AFTER the switch-back.
    releaseDial()
    await botActivation

    // The stale activation must not steal the foreground route (epoch guard).
    expect(activeGateway()).toBe(primary)

    // 4. No permanent break: switching to Bot again activates the (already
    // open) socket — no app restart, no duplicate socket/serve.
    await ensureGatewayForProfile('bot')

    expect(activeGateway()).toBe(gatewayMocks.instances[0])
    expect(gatewayMocks.instances[0].connectionState).toBe('open')
    expect(gatewayMocks.instances).toHaveLength(1)
  })
})

describe('secondary connection timeout (#93454)', () => {
  it("rejects instead of hanging forever when openSecondary's getConnection() wedges", async () => {
    // Repro: desktop.getConnection is an IPC round-trip into the main process
    // with no timeout of its own. A wedged main-process round-trip (e.g. a
    // stuck revalidation) hangs this await forever, latching
    // entry.connectPromise so every routed action against this secondary
    // (SSH terminal, messaging DELETE, session send, …) never settles either.
    vi.useFakeTimers()

    let callCount = 0
    const getConnection = vi.fn(({ profile }: { profile: string }) => {
      callCount += 1

      // First call is sharedPrimaryRoute's probe — resolves fast, not the
      // shared primary. Every call after (openSecondary's actual dial) wedges.
      if (callCount === 1) {
        return Promise.resolve({ sharedPrimary: false })
      }

      return new Promise(() => undefined)
    })

    installDesktop({ getConnection })

    const pending = expect(ensureGatewayForProfile('work')).rejects.toThrow('Timed out connecting to profile "work"')

    // Advance past the internal reconnect-attempt timeout (20s) — the stalled
    // await must reject instead of hanging forever.
    await vi.advanceTimersByTimeAsync(20_000)
    await pending
  })

  it('does not let a wedged shared-primary-route probe block the secondary dial forever', async () => {
    // Same unbounded-IPC hazard as above, but for sharedPrimaryRoute's own
    // getConnection() probe, which runs BEFORE openSecondary on every route —
    // a wedge there must resolve to "not the shared primary" and fall through
    // to the ordinary secondary dial instead of hanging the whole route
    // decision forever.
    vi.useFakeTimers()

    let callCount = 0
    const getConnection = vi.fn(({ profile }: { profile: string }) => {
      callCount += 1

      if (callCount === 1) {
        return new Promise(() => undefined)
      }

      return Promise.resolve({
        authMode: 'token',
        baseUrl: `https://${profile}.invalid`,
        mode: 'local',
        profile,
        token: 'fake-test-token',
        wsUrl: `wss://${profile}.invalid/ws`
      })
    })

    installDesktop({ getConnection })

    // The #92434 pin above leaves gatewayMocks.connect latched on a
    // never-resolving mockImplementation (vi.clearAllMocks() clears calls, not
    // implementations). Restore the default resolving dial for this test.
    gatewayMocks.connect.mockImplementation(async () => undefined)

    const pending = ensureGatewayForProfile('work')

    await vi.advanceTimersByTimeAsync(20_000)
    await pending

    // The probe's own bound (not just openSecondary's) is what let this
    // resolve after a single 20s timeout instead of two stacked ones.
    expect(callCount).toBe(2)
    expect(activeGateway()).toBe(gatewayMocks.instances[0])
  })
})
