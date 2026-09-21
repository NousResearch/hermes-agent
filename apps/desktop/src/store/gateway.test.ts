import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { BACKEND_BOOT_WAIT_TIMEOUT_MS, RECONNECT_ATTEMPT_TIMEOUT_MS } from '@/lib/with-timeout'

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
  const instances: {
    close: ReturnType<typeof vi.fn>
    connectionState: string
    emitState: (state: string) => void
  }[] = []

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
    stateListener: ((state: string) => void) | null = null
    emitState = (state: string): void => {
      this.connectionState = state
      this.stateListener?.(state)
    }
    // Mirrors json-rpc-gateway: 'connecting' → 'open' on success, → 'error' on a refused dial.
    connect = async (wsUrl: string): Promise<void> => {
      this.emitState('connecting')

      try {
        await gatewayMocks.connect(wsUrl)
      } catch (error) {
        this.emitState('error')

        throw error
      }

      this.emitState('open')
    }
    onEvent = vi.fn(() => () => {})
    onState = vi.fn((listener: (state: string) => void) => {
      this.stateListener = listener

      return () => {
        this.stateListener = null
      }
    })
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
// Reconnect lazily loads this store. Keep the session boundary isolated here
// too: the real store imports the UI graph and depends on unmocked session atoms.
vi.mock('@/store/session-states', () => ({
  reconcileBusyStatesOnReconnect: vi.fn(),
  resetTileRuntimeBindings: vi.fn()
}))

const {
  activeGateway,
  closeSecondaryGateways,
  configureGatewayRegistry,
  dispatchPrimaryServerRequest,
  ensureGatewayForAgent,
  ensureGatewayForProfile,
  openGatewayForProfile,
  openGatewayForAgent,
  pruneSecondaryGateways,
  SECONDARY_MIN_LIFETIME_MS,
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
  vi.restoreAllMocks()
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

    const getConnection = vi.fn(
      async ({ profile }: { profile: string }, _options?: { priority?: 'foreground'; speculative?: boolean }) => ({
        authMode: 'token',
        baseUrl: `https://${profile}.invalid`,
        mode: 'local',
        profile,
        token: 'fake-test-token',
        wsUrl: `wss://${profile}.invalid/ws`
      })
    )

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
    // This route never became active, so recovery remains speculative and
    // cannot occupy a queued local-pool slot.
    expect(getConnection.mock.calls.at(-1)?.[1]).toEqual({ speculative: true })
  })

  it('uses foreground recovery when the selected profile loses its socket', async () => {
    vi.useFakeTimers()

    let failReconnect = false

    const getConnection = vi.fn(
      async ({ profile }: { profile: string }, _options?: { priority?: 'foreground'; speculative?: boolean }) => ({
        authMode: 'token',
        baseUrl: `https://${profile}.invalid`,
        mode: 'local',
        profile,
        token: 'fake-test-token',
        wsUrl: `wss://${profile}.invalid/ws`
      })
    )

    installDesktop({ getConnection })
    await ensureGatewayForProfile('work')
    ;(activeGateway() as unknown as { connectionState: string }).connectionState = 'closed'
    gatewayMocks.connect.mockImplementation(async () => {
      if (failReconnect) {
        throw new Error('backend unreachable')
      }
    })
    failReconnect = true

    await expect(ensureGatewayForProfile('work')).rejects.toThrow('backend unreachable')
    failReconnect = false
    await vi.runAllTimersAsync()

    expect(getConnection.mock.calls.at(-1)?.[1]).toEqual({ priority: 'foreground' })
  })

  it('retries foreground after it promotes a rejected speculative dial', async () => {
    let rejectSpeculative!: (reason?: unknown) => void

    const speculative = new Promise<never>((_resolve, reject) => {
      rejectSpeculative = reject
    })

    const connection = {
      authMode: 'token',
      baseUrl: 'https://work.invalid',
      mode: 'local',
      profile: 'work',
      token: 'fake-test-token',
      wsUrl: 'wss://work.invalid/ws'
    }

    const getConnection = vi
      .fn()
      // Background route probe: this is a dedicated secondary, not shared primary.
      .mockResolvedValueOnce({ ...connection, sharedPrimary: false })
      // Its speculative secondary dial is rejected after the user clicks.
      .mockImplementationOnce(() => speculative)
      // The foreground route probe stays dedicated.
      .mockResolvedValueOnce({ ...connection, sharedPrimary: false })
      // Main promotes the pool spawn, then the renderer redials the socket.
      .mockResolvedValueOnce(connection)
      .mockResolvedValueOnce(connection)

    installDesktop({ getConnection })

    const warming = openGatewayForProfile('work', { speculative: true })
    await vi.waitFor(() => expect(getConnection).toHaveBeenCalledTimes(2))

    const selecting = ensureGatewayForProfile('work')
    await vi.waitFor(() => expect(getConnection).toHaveBeenCalledTimes(4))
    rejectSpeculative(new Error('local pool is full'))

    await expect(warming).rejects.toThrow('local pool is full')
    await expect(selecting).resolves.toBeUndefined()
    expect(activeGateway()).toBe(gatewayMocks.instances[0])
    expect(gatewayMocks.instances[0].connectionState).toBe('open')
    expect(getConnection.mock.calls.slice(-2)).toEqual([
      ['work', { priority: 'foreground' }],
      ['work', { priority: 'foreground' }]
    ])
  })

  it('bounds a foreground promotion after the speculative dial is rejected', async () => {
    vi.useFakeTimers()

    let rejectSpeculative!: (reason?: unknown) => void

    const speculative = new Promise<never>((_resolve, reject) => {
      rejectSpeculative = reject
    })

    const stalledPromotion = new Promise<never>(() => {})

    const connection = {
      authMode: 'token',
      baseUrl: 'https://work.invalid',
      mode: 'local',
      profile: 'work',
      token: 'fake-test-token',
      wsUrl: 'wss://work.invalid/ws'
    }

    const getConnection = vi
      .fn()
      .mockResolvedValueOnce({ ...connection, sharedPrimary: false })
      .mockImplementationOnce(() => speculative)
      .mockResolvedValueOnce({ ...connection, sharedPrimary: false })
      .mockImplementationOnce(() => stalledPromotion)

    installDesktop({ getConnection })

    const warming = openGatewayForProfile('work', { speculative: true })

    const warmingFailure = warming.then(
      () => undefined,
      error => error
    )

    await vi.waitFor(() => expect(getConnection).toHaveBeenCalledTimes(2))

    const selecting = ensureGatewayForProfile('work')

    const selectingFailure = selecting.then(
      () => undefined,
      error => error
    )

    await vi.waitFor(() => expect(getConnection).toHaveBeenCalledTimes(4))
    rejectSpeculative(new Error('local pool is full'))

    let selectionSettled = false
    void selectingFailure.then(() => {
      selectionSettled = true
    })

    await vi.advanceTimersByTimeAsync(RECONNECT_ATTEMPT_TIMEOUT_MS)

    // Promotion is part of the explicit may-spawn activation, so the ordinary
    // reconnect limit must not reject it while main can still be cold-starting
    // the backend.
    expect(selectionSettled).toBe(false)

    await vi.advanceTimersByTimeAsync(BACKEND_BOOT_WAIT_TIMEOUT_MS - RECONNECT_ATTEMPT_TIMEOUT_MS)

    await expect(warmingFailure).resolves.toMatchObject({ message: 'local pool is full' })
    await expect(selectingFailure).resolves.toMatchObject({ message: 'local pool is full' })
  })

  it('handles a rejected foreground promotion before the speculative dial settles', async () => {
    let rejectSpeculative!: (reason?: unknown) => void

    const speculative = new Promise<never>((_resolve, reject) => {
      rejectSpeculative = reject
    })

    const connection = {
      authMode: 'token',
      baseUrl: 'https://work.invalid',
      mode: 'local',
      profile: 'work',
      token: 'fake-test-token',
      wsUrl: 'wss://work.invalid/ws'
    }

    const getConnection = vi
      .fn()
      .mockResolvedValueOnce({ ...connection, sharedPrimary: false })
      .mockImplementationOnce(() => speculative)
      .mockResolvedValueOnce({ ...connection, sharedPrimary: false })
      .mockRejectedValueOnce(new Error('foreground promotion refused'))

    installDesktop({ getConnection })

    const warming = openGatewayForProfile('work', { speculative: true })

    const warmingFailure = warming.then(
      () => undefined,
      error => error
    )

    await vi.waitFor(() => expect(getConnection).toHaveBeenCalledTimes(2))

    const selecting = ensureGatewayForProfile('work')

    const selectingFailure = selecting.then(
      () => undefined,
      error => error
    )

    await vi.waitFor(() => expect(getConnection).toHaveBeenCalledTimes(4))

    // The foreground request rejects before hydration gives up. Its rejection
    // must already be observed; the later original failure remains the one
    // surfaced to the user.
    await new Promise(resolve => setTimeout(resolve, 0))
    rejectSpeculative(new Error('local pool is full'))

    await expect(warmingFailure).resolves.toMatchObject({ message: 'local pool is full' })
    await expect(selectingFailure).resolves.toMatchObject({ message: 'local pool is full' })
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

      const calls = errorSpy.mock.calls
      const messages = calls.map(([message]) => String(message))
      const contexts = calls.map(([, context]) => context as { scope?: string; profile?: string })

      expect(messages).toHaveLength(2)
      expect(messages.every(message => message === '[gateway] dial failed')).toBe(true)
      expect(contexts).toEqual(
        expect.arrayContaining([
          expect.objectContaining({ scope: 'conn:work::default', profile: 'default' }),
          expect.objectContaining({ scope: 'conn:homelab::default', profile: 'default' })
        ])
      )
      expect(new Set(contexts.map(context => context.scope)).size).toBe(2)

      for (const [, , error] of calls) {
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

    const pending = expect(openGatewayForProfile('work')).rejects.toThrow('Timed out connecting to profile "work"')

    // A background/prewarm dial remains an ordinary reconnect-class attempt:
    // it must reject after 20s instead of inheriting the foreground boot wait.
    await vi.advanceTimersByTimeAsync(RECONNECT_ATTEMPT_TIMEOUT_MS)
    await pending
  })

  it('does not let a wedged background route probe block the secondary dial forever', async () => {
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

    const pending = openGatewayForProfile('work')

    await vi.advanceTimersByTimeAsync(RECONNECT_ATTEMPT_TIMEOUT_MS)
    await pending

    // Background route probing remains reconnect-class: after its 20s bound,
    // it falls through to the immediately available secondary descriptor.
    expect(callCount).toBe(2)
    expect(activeGateway()).not.toBe(gatewayMocks.instances[0])
    expect(gatewayMocks.instances[0].connectionState).toBe('open')
  })

  it('uses one cold-activation deadline across route fallback, pruning, failure, and retry', async () => {
    vi.useFakeTimers()
    gatewayMocks.connect.mockImplementation(async () => undefined)

    const connection = {
      authMode: 'token',
      baseUrl: 'https://work.invalid',
      mode: 'local',
      profile: 'work',
      token: 'fake-test-token',
      wsUrl: 'wss://work.invalid/ws'
    }

    let callCount = 0

    const getConnection = vi.fn(() => {
      callCount += 1

      // Establish renderer history, then prune it. A later foreground reopen
      // may still respawn a dead child and must be treated as may-spawn.
      if (callCount === 1) {
        return Promise.resolve({ ...connection, sharedPrimary: false })
      }

      if (callCount === 2) {
        return Promise.resolve(connection)
      }

      // The first reopen succeeds after 45s total: both route resolution and
      // the secondary's main-process dial exceed the 20s reconnect budget.
      if (callCount === 3) {
        return new Promise(resolve => {
          setTimeout(() => resolve({ ...connection, sharedPrimary: false }), 25_000)
        })
      }

      if (callCount === 4) {
        return new Promise(resolve => {
          setTimeout(() => resolve(connection), 20_000)
        })
      }

      // The next activation proves the route fallback cannot reset the total
      // budget: it leaves only 1s for the secondary dial.
      if (callCount === 5) {
        return new Promise(resolve => {
          setTimeout(() => resolve({ ...connection, sharedPrimary: false }), BACKEND_BOOT_WAIT_TIMEOUT_MS - 1_000)
        })
      }

      if (callCount === 6) {
        return new Promise(() => undefined)
      }

      return Promise.resolve(callCount === 7 ? { ...connection, sharedPrimary: false } : connection)
    })

    installDesktop({ getConnection })

    await openGatewayForProfile('work')
    await vi.advanceTimersByTimeAsync(SECONDARY_MIN_LIFETIME_MS + 1)
    pruneSecondaryGateways(new Set())
    expect(gatewayMocks.instances[0].close).toHaveBeenCalledTimes(1)

    const reopening = ensureGatewayForProfile('work')

    await vi.advanceTimersByTimeAsync(31_000)
    pruneSecondaryGateways(new Set())
    expect(gatewayMocks.instances[1].close).not.toHaveBeenCalled()

    await vi.advanceTimersByTimeAsync(14_000)
    await expect(reopening).resolves.toBeUndefined()
    expect(activeGateway()).toBe(gatewayMocks.instances[1])

    await ensureGatewayForProfile('default')
    await vi.advanceTimersByTimeAsync(SECONDARY_MIN_LIFETIME_MS + 1)
    pruneSecondaryGateways(new Set())
    expect(gatewayMocks.instances[1].close).toHaveBeenCalledTimes(1)

    const timingOut = ensureGatewayForProfile('work')
    let timedOut = false

    const observedTimeout = timingOut.catch(error => {
      timedOut = true

      return error
    })

    await vi.advanceTimersByTimeAsync(BACKEND_BOOT_WAIT_TIMEOUT_MS - 1)
    expect(timedOut).toBe(false)

    await vi.advanceTimersByTimeAsync(1)
    await expect(observedTimeout).resolves.toMatchObject({ message: 'Timed out connecting to profile "work"' })
    expect(activeGateway()).not.toBe(gatewayMocks.instances[2])

    // Settlement releases this activation's lease, so pruning can reclaim the
    // failed entry and a fresh explicit attempt gets a fresh bounded budget.
    pruneSecondaryGateways(new Set())
    expect(gatewayMocks.instances[2].close).toHaveBeenCalledTimes(1)

    await expect(ensureGatewayForProfile('work')).resolves.toBeUndefined()
    expect(activeGateway()).toBe(gatewayMocks.instances[3])
  })

  it('does not start a fallback IPC after the cold-activation deadline expires', async () => {
    vi.useFakeTimers()
    gatewayMocks.connect.mockImplementation(async () => undefined)

    const getConnection = vi.fn(
      ({ profile }: { profile: string }) =>
        new Promise(resolve => {
          setTimeout(() => resolve({ profile, sharedPrimary: false }), BACKEND_BOOT_WAIT_TIMEOUT_MS)
        })
    )

    installDesktop({ getConnection })
    const primary = activeGateway()
    const observed = ensureGatewayForProfile('work').catch(error => error)

    await vi.advanceTimersByTimeAsync(BACKEND_BOOT_WAIT_TIMEOUT_MS)

    await expect(observed).resolves.toMatchObject({ message: 'Timed out connecting to profile "work"' })
    expect(getConnection).toHaveBeenCalledTimes(1)
    expect(gatewayMocks.instances).toHaveLength(0)
    expect(activeGateway()).toBe(primary)
  })

  it('does not publish a shared-primary result that resolves at the activation deadline', async () => {
    vi.useFakeTimers()

    const getConnection = vi.fn(({ profile }: { profile: string }) => {
      vi.setSystemTime(Date.now() + BACKEND_BOOT_WAIT_TIMEOUT_MS)

      return Promise.resolve({ profile, sharedPrimary: true })
    })

    installDesktop({ getConnection })
    const primary = activeGateway()

    await expect(ensureGatewayForProfile('work')).rejects.toThrow('Timed out connecting to profile "work"')

    expect(getConnection).toHaveBeenCalledTimes(1)
    expect(gatewayMocks.instances).toHaveLength(0)
    expect(activeGateway()).toBe(primary)
  })

  it('does not publish a secondary whose WebSocket opens at the activation deadline', async () => {
    vi.useFakeTimers()

    const connection = {
      authMode: 'token',
      baseUrl: 'https://work.invalid',
      mode: 'local',
      profile: 'work',
      token: 'fake-test-token',
      wsUrl: 'wss://work.invalid/ws'
    }

    let callCount = 0

    const getConnection = vi.fn(() => {
      callCount += 1

      return Promise.resolve(callCount === 1 ? { ...connection, sharedPrimary: false } : connection)
    })

    installDesktop({ getConnection })
    const primary = activeGateway()

    gatewayMocks.connect.mockImplementation(async () => {
      vi.setSystemTime(Date.now() + BACKEND_BOOT_WAIT_TIMEOUT_MS)
    })

    await expect(ensureGatewayForProfile('work')).rejects.toThrow(
      'Timed out connecting the gateway WebSocket for profile "work"'
    )

    expect(getConnection).toHaveBeenCalledTimes(2)
    expect(gatewayMocks.instances).toHaveLength(1)
    expect(gatewayMocks.instances[0].close).toHaveBeenCalledTimes(1)
    expect(activeGateway()).toBe(primary)
  })

  it('does not let an older same-scope foreground open steal a newer activation lease', async () => {
    vi.useFakeTimers()
    gatewayMocks.connect.mockImplementation(async () => undefined)

    const connection = {
      authMode: 'token',
      baseUrl: 'https://work.invalid',
      mode: 'local',
      profile: 'work',
      token: 'fake-test-token',
      wsUrl: 'wss://work.invalid/ws'
    }

    let resolveOlderRoute!: (result: typeof connection & { sharedPrimary?: boolean }) => void
    let resolveNewerDial!: (result: typeof connection) => void
    let callCount = 0

    const getConnection = vi.fn(() => {
      callCount += 1

      if (callCount === 1) {
        return Promise.resolve({ ...connection, sharedPrimary: false })
      }

      if (callCount === 2) {
        return Promise.resolve(connection)
      }

      if (callCount === 3) {
        return new Promise<typeof connection>(resolve => {
          resolveOlderRoute = resolve
        })
      }

      if (callCount === 4) {
        return Promise.resolve({ ...connection, sharedPrimary: false })
      }

      if (callCount === 5) {
        return new Promise<typeof connection>(resolve => {
          resolveNewerDial = resolve
        })
      }

      return Promise.resolve(connection)
    })

    installDesktop({ getConnection })
    await openGatewayForProfile('work')
    gatewayMocks.instances[0].connectionState = 'closed'

    const older = openGatewayForProfile('work', { spawnPriority: 'foreground' })

    const olderResult = older.then(
      () => undefined,
      error => error
    )

    await vi.waitFor(() => expect(getConnection).toHaveBeenCalledTimes(3))

    await vi.advanceTimersByTimeAsync(10_000)
    const newer = openGatewayForProfile('work', { spawnPriority: 'foreground' })
    await vi.waitFor(() => expect(getConnection).toHaveBeenCalledTimes(5))
    resolveOlderRoute({ ...connection, sharedPrimary: false })

    // The older deadline has now elapsed, but the newer activation still has
    // 9s and owns the only in-flight dial. Its lease must survive this prune.
    await vi.advanceTimersByTimeAsync(BACKEND_BOOT_WAIT_TIMEOUT_MS - 9_000)
    pruneSecondaryGateways(new Set())
    expect(getConnection).toHaveBeenCalledTimes(5)
    expect(gatewayMocks.instances[0].close).not.toHaveBeenCalled()

    resolveNewerDial(connection)
    await expect(newer).resolves.toBeUndefined()
    await expect(olderResult).resolves.toMatchObject({ message: 'Gateway activation superseded for profile "work"' })

    await vi.advanceTimersByTimeAsync(SECONDARY_MIN_LIFETIME_MS + 1)
    pruneSecondaryGateways(new Set())
    expect(gatewayMocks.instances[0].close).toHaveBeenCalledTimes(1)
  })

  it('carries a two-phase agent activation lease through open then ensure', async () => {
    vi.useFakeTimers()
    gatewayMocks.connect.mockImplementation(async () => undefined)

    const connection = {
      authMode: 'token',
      connectionId: 'homelab',
      mode: 'local',
      profile: 'research',
      token: 'fake-test-token',
      wsUrl: 'wss://homelab.invalid/ws'
    }

    const getConnectionFor = vi.fn(
      () =>
        new Promise(resolve => {
          setTimeout(() => resolve({ ...connection, sharedRemote: false }), 45_000)
        })
    )

    installDesktop({ getConnectionFor })
    const activationController = new AbortController()

    const opening = openGatewayForAgent('homelab', 'research', {
      activationLease: true,
      signal: activationController.signal,
      spawnPriority: 'foreground'
    })

    await vi.advanceTimersByTimeAsync(50_000)
    pruneSecondaryGateways(new Set())
    expect(gatewayMocks.instances).toHaveLength(1)
    expect(gatewayMocks.instances[0].close).not.toHaveBeenCalled()

    await vi.advanceTimersByTimeAsync(40_000)
    await expect(opening).resolves.toBeUndefined()

    // The successful prepare keeps its owned lease for the synchronous commit
    // phase; ensure consumes that handoff and releases only its own lease.
    pruneSecondaryGateways(new Set())
    expect(gatewayMocks.instances[0].close).not.toHaveBeenCalled()
    await expect(ensureGatewayForAgent('homelab', 'research', { signal: activationController.signal })).resolves.toBe(
      true
    )

    await ensureGatewayForProfile('default')
    pruneSecondaryGateways(new Set())
    expect(gatewayMocks.instances[0].close).toHaveBeenCalledTimes(1)
  })

  it('does not reset an expired two-phase activation budget during ensure', async () => {
    vi.useFakeTimers()
    gatewayMocks.connect.mockImplementation(async () => undefined)

    const connection = {
      authMode: 'token',
      connectionId: 'homelab',
      mode: 'local',
      profile: 'research',
      token: 'fake-test-token',
      wsUrl: 'wss://homelab.invalid/ws'
    }

    let callCount = 0

    const getConnectionFor = vi.fn(() => {
      callCount += 1

      if (callCount > 1) {
        return Promise.resolve(connection)
      }

      return new Promise(resolve => {
        setTimeout(() => resolve(connection), BACKEND_BOOT_WAIT_TIMEOUT_MS - 1)
      })
    })

    installDesktop({ getConnectionFor })
    const activationController = new AbortController()

    const opening = openGatewayForAgent('homelab', 'research', {
      activationLease: true,
      signal: activationController.signal,
      spawnPriority: 'foreground'
    })

    await vi.advanceTimersByTimeAsync(BACKEND_BOOT_WAIT_TIMEOUT_MS - 1)
    await expect(opening).resolves.toBeUndefined()
    await vi.advanceTimersByTimeAsync(1)

    // The entry may be reclaimed between the two phases. Correlation must
    // still retain this transaction's expired deadline without redialing.
    pruneSecondaryGateways(new Set())

    await expect(ensureGatewayForAgent('homelab', 'research', { signal: activationController.signal })).resolves.toBe(
      false
    )
    expect(getConnectionFor).toHaveBeenCalledTimes(1)
    expect(gatewayMocks.instances).toHaveLength(1)
  })

  it('gives a fresh standalone ensure its own budget instead of claiming an orphaned handoff', async () => {
    vi.useFakeTimers()
    gatewayMocks.connect.mockImplementation(async () => undefined)

    const connection = {
      authMode: 'token',
      connectionId: 'homelab',
      mode: 'local',
      profile: 'research',
      token: 'fake-test-token',
      wsUrl: 'wss://homelab.invalid/ws'
    }

    const getConnectionFor = vi.fn(
      () =>
        new Promise(resolve => {
          setTimeout(() => resolve(connection), BACKEND_BOOT_WAIT_TIMEOUT_MS - 1)
        })
    )

    installDesktop({ getConnectionFor })
    const abandonedController = new AbortController()

    const opening = openGatewayForAgent('homelab', 'research', {
      activationLease: true,
      signal: abandonedController.signal,
      spawnPriority: 'foreground'
    })

    await vi.advanceTimersByTimeAsync(BACKEND_BOOT_WAIT_TIMEOUT_MS - 1)
    await expect(opening).resolves.toBeUndefined()
    await vi.advanceTimersByTimeAsync(1)

    // This is an unrelated SDK/profile intent: without the source
    // transaction's signal it must not inherit that abandoned deadline.
    await expect(ensureGatewayForAgent('homelab', 'research')).resolves.toBe(true)
    expect(getConnectionFor).toHaveBeenCalledTimes(1)

    pruneSecondaryGateways(new Set())
    expect(gatewayMocks.instances[0].close).not.toHaveBeenCalled()
  })
})

describe('server→client request routing without a registry handler (#112791)', () => {
  it('answers -32601 when the registry has no onServerRequest, and forwards with the profile when it does', () => {
    const request = { fail: vi.fn(), id: 'srq-1', method: 'clarify', params: { session_id: 's1' }, respond: vi.fn() }

    // beforeEach configured a registry with onEvent only: nobody can answer,
    // so the handler declines (false) and the channel answers -32601 now
    // instead of the backend waiting out its deadline.
    expect(dispatchPrimaryServerRequest(request as never, 'default')).toBe(false)
    expect(request.fail).not.toHaveBeenCalled()

    const onServerRequest = vi.fn()

    configureGatewayRegistry({ onEvent: vi.fn(), onServerRequest } as never)
    expect(dispatchPrimaryServerRequest(request as never, 'work')).toBe(true)
    expect(request.fail).not.toHaveBeenCalled()
    expect(onServerRequest).toHaveBeenCalledTimes(1)
    expect(onServerRequest).toHaveBeenCalledWith(
      expect.objectContaining({ id: 'srq-1', method: 'clarify', profile: 'work' })
    )
  })
})

describe('secondary reconnect backoff (#83134)', () => {
  const installWork = (): void =>
    installDesktop({
      getConnection: vi.fn(async ({ profile }: { profile: string }) => ({
        authMode: 'token',
        baseUrl: `https://${profile}.invalid`,
        mode: 'local',
        profile,
        token: 'fake-test-token',
        wsUrl: `wss://${profile}.invalid/ws`
      }))
    })

  it('climbs the ladder while the backend stays down after a stable session', async () => {
    vi.useFakeTimers()
    vi.spyOn(Math, 'random').mockReturnValue(0.5) // deterministic ladder: 150/300/600/…
    installWork()

    await ensureGatewayForProfile('work')
    const socket = gatewayMocks.instances[0]

    // A session that lived well past the stable-open window, then the backend dies.
    await vi.advanceTimersByTimeAsync(6_000)
    gatewayMocks.connect.mockRejectedValue(new Error('connection refused'))
    socket.emitState('closed')

    // Every redial is refused ('connecting' → 'error'). A refused dial never
    // opened, so it must not look like a stable session and reset the ladder.
    await vi.advanceTimersByTimeAsync(12_000)

    // Ladder from attempt 0: ≤ ~8 dials in 12 s; a reset-per-failure loop makes ~40+.
    // Deterministic ladder 150/300/…/4800 ms ⇒ 6–7 redials in 12 s: alive, but climbing.
    expect(gatewayMocks.connect.mock.calls.length).toBeGreaterThanOrEqual(5)
    expect(gatewayMocks.connect.mock.calls.length).toBeLessThanOrEqual(10)
  })

  it('treats an accept-then-close socket as a failed attempt', async () => {
    vi.useFakeTimers()
    vi.spyOn(Math, 'random').mockReturnValue(0.5)
    installWork()
    gatewayMocks.connect.mockResolvedValue(undefined)

    await ensureGatewayForProfile('work')

    // Each redial "succeeds" and the socket dies 100 ms later, forever.
    gatewayMocks.connect.mockImplementation(async () => {
      const socket = gatewayMocks.instances.at(-1)!
      setTimeout(() => socket.emitState('closed'), 100)
    })
    gatewayMocks.instances[0].emitState('closed')

    await vi.advanceTimersByTimeAsync(12_000)

    // Deterministic ladder 150/300/…/4800 ms ⇒ 6–7 redials in 12 s: alive, but climbing.
    expect(gatewayMocks.connect.mock.calls.length).toBeGreaterThanOrEqual(5)
    expect(gatewayMocks.connect.mock.calls.length).toBeLessThanOrEqual(10)
  })
})
