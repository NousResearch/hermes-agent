import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// Regression coverage for the #89206 wake-failure class: session-scoped RPCs
// routed to a backend that does not own the session's profile. Three layers:
//   1. The registry publishes the ACTIVE route's profile ($activeGatewayRoute)
//      from applyActive itself, so eviction fallbacks move it in lockstep.
//   2. store/profile.ts mirrors that atom into $activeGatewayProfile, so the
//      "already active" fast path can never trust a stale profile.
//   3. session-request-router pins session-scoped RPCs to the owning
//      profile's socket at REQUEST time when the active route diverges.

const secondaryGateways: Array<{
  close: ReturnType<typeof vi.fn>
  connect: ReturnType<typeof vi.fn>
  connectionState: string
  emit: (event: { payload?: Record<string, unknown>; session_id?: string; type: string }) => void
  emitState: (state: string) => void
  request: ReturnType<typeof vi.fn>
}> = []

let promptAckStatus: null | string = null

vi.mock('@/hermes', () => ({
  HermesGateway: class {
    connectionState = 'closed'
    eventHandler: ((event: { payload?: Record<string, unknown>; session_id?: string; type: string }) => void) | null = null
    stateHandler: ((state: string) => void) | null = null
    connect = vi.fn(async () => {
      this.connectionState = 'open'
    })
    request = vi.fn(async (method: string, params: Record<string, unknown>) => {
      if (this.connectionState !== 'open') {
        throw new Error('gateway is not connected')
      }

      return method === 'prompt.submit' && promptAckStatus ? { status: promptAckStatus } : { method, params }
    })
    close = vi.fn()
    emit = (event: { payload?: Record<string, unknown>; session_id?: string; type: string }) => this.eventHandler?.(event)
    emitState = (state: string) => this.stateHandler?.(state)
    onEvent = vi.fn((handler: (event: { payload?: Record<string, unknown>; session_id?: string; type: string }) => void) => {
      this.eventHandler = handler

      return () => {
        this.eventHandler = null
      }
    })
    onState = vi.fn((handler: (state: string) => void) => {
      this.stateHandler = handler

      return () => {
        this.stateHandler = null
      }
    })

    constructor() {
      secondaryGateways.push(this)
    }
  },
  setApiRequestConnection: vi.fn()
}))
vi.mock('@/store/session', () => ({ setConnection: vi.fn(), setGatewayState: vi.fn() }))
vi.mock('@/store/notify-baseline', () => ({ markNativeNotifyBaseline: vi.fn() }))

const {
  $activeGatewayRoute,
  activeGatewayProfileKey,
  closeSecondaryGateways,
  configureGatewayRegistry,
  ensureGatewayForProfile,
  pruneSecondaryGateways,
  retireLocalProfileGateways,
  setPrimaryGateway
} = await import('./gateway')

const { requestForSessionProfile, sessionRpcNeedsProfileRoute } = await import('./session-request-router')

function installDesktop(): void {
  ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = {
    getConnection: vi.fn(async (profile: null | string) =>
      profile ? { port: 5151, profile, token: 'secondary-token' } : { port: 4242, token: 'primary-token' }
    ),
    getConnectionFor: vi.fn(async ({ connectionId, profile }: { connectionId: string; profile: string }) => ({
      port: connectionId === 'source-a' ? 6161 : 6262,
      profile,
      token: `${connectionId}-token`
    })),
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
  promptAckStatus = null
  configureGatewayRegistry({ onEvent: vi.fn() })
  closeSecondaryGateways()
})

afterEach(() => {
  closeSecondaryGateways()
  vi.clearAllMocks()
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
})

describe('$activeGatewayRoute (registry-owned active profile)', () => {
  it('tracks profile activation and eviction fallback in lockstep with the socket', async () => {
    const primary = makePrimary()
    setPrimaryGateway(primary as never, 'default')
    installDesktop()

    await ensureGatewayForProfile('default')
    expect(activeGatewayProfileKey()).toBe('default')

    await ensureGatewayForProfile('loki')
    expect(activeGatewayProfileKey()).toBe('loki')
    expect($activeGatewayRoute.get()).toBe('loki')

    // Idle-reap style eviction of everything but... nothing keeps loki alive.
    // The registry must move BOTH the socket and the published profile back
    // to the primary — before the fix only the socket moved, and the stale
    // profile atom made ensureGatewayProfile skip the re-swap forever.
    retireLocalProfileGateways('loki')
    expect(activeGatewayProfileKey()).toBe('default')
    expect($activeGatewayRoute.get()).toBe('default')
  })

  it('falls back to primary when pruning evicts the active secondary', async () => {
    const primary = makePrimary()
    setPrimaryGateway(primary as never, 'default')
    installDesktop()

    await ensureGatewayForProfile('hulk')
    expect(activeGatewayProfileKey()).toBe('hulk')

    // Force-evict the active entry (retention flags off) — the keep-set is
    // empty and the active guard is bypassed by retiring first.
    retireLocalProfileGateways('hulk')
    pruneSecondaryGateways(new Set())

    expect(activeGatewayProfileKey()).toBe('default')
  })
})

describe('sessionRpcNeedsProfileRoute', () => {
  it('routes ambient ONLY when the owner is unknown (no session / global chrome)', () => {
    expect(sessionRpcNeedsProfileRoute(null)).toBe(false)
    expect(sessionRpcNeedsProfileRoute(undefined)).toBe(false)
    expect(sessionRpcNeedsProfileRoute('')).toBe(false)
    expect(sessionRpcNeedsProfileRoute('   ')).toBe(false)
  })

  it('pins a KNOWN owner to its own profile regardless of what is active', () => {
    // No active-profile comparison exists anymore: "active" is presentation
    // state, never a routing authority. A known owner ALWAYS routes to its own
    // profile — even when it happens to equal whatever is currently active,
    // gatewayForProfile collapses that back to the primary socket (no cost).
    expect(sessionRpcNeedsProfileRoute('loki')).toBe(true)
    expect(sessionRpcNeedsProfileRoute('default')).toBe(true)
    expect(sessionRpcNeedsProfileRoute('hulk')).toBe(true)
  })

  it('pins a route owner with a connectionId', () => {
    expect(sessionRpcNeedsProfileRoute({ connectionId: 'local', profile: 'developer' })).toBe(true)
    expect(sessionRpcNeedsProfileRoute({ connectionId: '', profile: 'developer' })).toBe(false)
  })
})

describe('requestForSessionProfile', () => {
  it('keeps concurrent same-name requests pinned while foreground activation changes', async () => {
    const primary = makePrimary()
    setPrimaryGateway(primary as never, 'default')
    installDesktop()
    await ensureGatewayForProfile('other')

    const desktop = (
      window as unknown as {
        hermesDesktop: { getConnectionFor: ReturnType<typeof vi.fn> }
      }
    ).hermesDesktop

    const ambient = vi.fn(async () => ({ ambient: true }))

    const routeA = {
      connectionId: 'source-a',
      profile: 'default',
      targetProfile: 'backend-a'
    }

    const routeB = {
      connectionId: 'source-b',
      profile: 'default',
      targetProfile: 'backend-b'
    }

    const fromA = requestForSessionProfile(routeA, ambient as never, 'session.resume', {
      profile: 'default',
      session_id: 'stored-a'
    })

    routeA.connectionId = 'source-b'
    routeA.targetProfile = 'mutated-after-dispatch'
    await ensureGatewayForProfile('default')

    const fromB = requestForSessionProfile(routeB, ambient as never, 'session.resume', {
      profile: 'default',
      session_id: 'stored-b'
    })

    await Promise.all([fromA, fromB])

    expect(desktop.getConnectionFor).toHaveBeenCalledWith({ connectionId: 'source-a', profile: 'default' })
    expect(desktop.getConnectionFor).toHaveBeenCalledWith({ connectionId: 'source-b', profile: 'default' })
    expect(secondaryGateways).toHaveLength(3)
    expect(secondaryGateways[1].request).toHaveBeenCalledWith('session.resume', {
      profile: 'backend-a',
      session_id: 'stored-a'
    })
    expect(secondaryGateways[2].request).toHaveBeenCalledWith('session.resume', {
      profile: 'backend-b',
      session_id: 'stored-b'
    })
    expect(ambient).not.toHaveBeenCalled()
  })

  it('rejects an explicit route without a connection instead of using ambient state', async () => {
    const ambient = vi.fn(async () => ({ ambient: true }))

    await expect(
      requestForSessionProfile(
        { connectionId: '', profile: 'default', targetProfile: 'backend-default' },
        ambient as never,
        'session.resume',
        { session_id: 'stored-a' }
      )
    ).rejects.toThrow(/missing connectionId/i)
    expect(ambient).not.toHaveBeenCalled()
  })

  it("dispatches on the owning profile's own socket when the active route moved off it (#89206)", async () => {
    const primary = makePrimary()
    setPrimaryGateway(primary as never, 'default')
    installDesktop()
    await ensureGatewayForProfile('default')

    const ambient = vi.fn(async (method: string, params?: Record<string, unknown>) => ({
      ambient: true,
      method,
      params
    }))

    // Active route is 'default'; the session belongs to 'loki'. The failing
    // path sent session.resume on the ambient (default) socket — the default
    // backend has never heard of the session and the bot never woke.
    const result = await requestForSessionProfile<{ method: string; params: Record<string, unknown> }>(
      'loki',
      ambient as never,
      'session.resume',
      { session_id: 'stored-loki-chat' }
    )

    expect(ambient).not.toHaveBeenCalled()
    expect(result).toEqual({ method: 'session.resume', params: { session_id: 'stored-loki-chat' } })
    expect(secondaryGateways).toHaveLength(1)
    expect(secondaryGateways[0].request).toHaveBeenCalledWith('session.resume', { session_id: 'stored-loki-chat' })
  })

  it('forwards timeout and abort signal onto the owning profile socket', async () => {
    const primary = makePrimary()
    setPrimaryGateway(primary as never, 'default')
    installDesktop()
    await ensureGatewayForProfile('default')

    const ambient = vi.fn(async (method: string, params?: Record<string, unknown>) => ({
      ambient: true,
      method,
      params
    }))

    const controller = new AbortController()

    await requestForSessionProfile(
      'loki',
      ambient as never,
      'prompt.submit',
      { session_id: 'stored-loki-chat', text: 'hi' },
      1_800_000,
      controller.signal
    )

    expect(ambient).not.toHaveBeenCalled()
    expect(secondaryGateways[0].request).toHaveBeenCalledWith(
      'prompt.submit',
      { session_id: 'stored-loki-chat', text: 'hi' },
      1_800_000,
      controller.signal
    )
  })

  it('forwards timeout and abort signal onto the owning connection socket', async () => {
    const primary = makePrimary()
    setPrimaryGateway(primary as never, 'default')
    installDesktop()
    const ambient = vi.fn(async () => ({ ambient: true }))
    const controller = new AbortController()

    await requestForSessionProfile(
      {
        connectionId: 'source-a',
        profile: 'default',
        targetProfile: 'backend-default'
      },
      ambient as never,
      'prompt.submit',
      { profile: 'default', session_id: 'stored-remote-chat', text: 'hi' },
      1_800_000,
      controller.signal
    )

    expect(ambient).not.toHaveBeenCalled()
    expect(secondaryGateways[0].request).toHaveBeenCalledWith(
      'prompt.submit',
      { profile: 'backend-default', session_id: 'stored-remote-chat', text: 'hi' },
      1_800_000,
      controller.signal
    )
  })

  it('keeps a routed prompt socket alive until the turn terminal event (#client-gone)', async () => {
    const primary = makePrimary()
    setPrimaryGateway(primary as never, 'default')
    installDesktop()
    const ambient = vi.fn(async () => ({ ambient: true }))

    await requestForSessionProfile(
      { connectionId: 'local', profile: 'default' },
      ambient as never,
      'prompt.submit',
      { session_id: 'rt-bot-chat', text: 'research this' }
    )

    // prompt.submit ACKs immediately while the model keeps running. Releasing
    // the request-scoped socket here detaches the runtime session; the backend
    // then interrupts it on the 20-second client-gone timer.
    expect(secondaryGateways[0].close).not.toHaveBeenCalled()

    vi.useFakeTimers()

    secondaryGateways[0].emit({
      payload: { running: false },
      session_id: 'rt-bot-chat',
      type: 'session.info'
    })

    // A chained goal/queue turn starts immediately after the settled frame;
    // it must cancel the pending release and inherit the live socket.
    secondaryGateways[0].emit({ session_id: 'rt-bot-chat', type: 'message.start' })
    await vi.advanceTimersByTimeAsync(500)
    expect(secondaryGateways[0].close).not.toHaveBeenCalled()

    secondaryGateways[0].emit({
      payload: { running: false },
      session_id: 'rt-bot-chat',
      type: 'session.info'
    })
    await vi.advanceTimersByTimeAsync(500)

    expect(secondaryGateways[0].close).toHaveBeenCalledOnce()
    vi.useRealTimers()
  })

  it.each(['queued', 'redirected', 'future-nonterminal'])('retains a routed socket for non-terminal ACK status %s', async status => {
    const primary = makePrimary()
    setPrimaryGateway(primary as never, 'default')
    installDesktop()
    const ambient = vi.fn(async () => ({ ambient: true }))

    promptAckStatus = status

    await requestForSessionProfile(
      'loki',
      ambient as never,
      'prompt.submit',
      { session_id: `rt-${status}`, text: 'continue' }
    )

    expect(secondaryGateways[0].close).not.toHaveBeenCalled()
  })

  it.each(['complete', 'completed', 'error'])('releases a routed socket for terminal ACK status %s', async status => {
    const primary = makePrimary()
    setPrimaryGateway(primary as never, 'default')
    installDesktop()
    const ambient = vi.fn(async () => ({ ambient: true }))

    promptAckStatus = status

    await requestForSessionProfile(
      'loki',
      ambient as never,
      'prompt.submit',
      { session_id: `rt-${status}`, text: 'finish' }
    )

    expect(secondaryGateways[0].close).toHaveBeenCalledOnce()
  })

  it('releases turn leases when their route is pruned and creates a fresh route next time', async () => {
    const primary = makePrimary()
    setPrimaryGateway(primary as never, 'default')
    installDesktop()
    const ambient = vi.fn(async () => ({ ambient: true }))
    const route = { connectionId: 'source-a', profile: 'worker' }

    await requestForSessionProfile(route, ambient as never, 'prompt.submit', {
      session_id: 'rt-pruned',
      text: 'work'
    })
    expect(secondaryGateways[0].close).not.toHaveBeenCalled()

    pruneSecondaryGateways(new Set())
    expect(secondaryGateways[0].close).toHaveBeenCalledOnce()

    await requestForSessionProfile(route, ambient as never, 'session.resume', { session_id: 'rt-pruned' })
    expect(secondaryGateways).toHaveLength(2)
  })

  it('releases a retained turn when its owning socket closes without a terminal event', async () => {
    const primary = makePrimary()
    setPrimaryGateway(primary as never, 'default')
    installDesktop()
    const ambient = vi.fn(async () => ({ ambient: true }))

    await requestForSessionProfile('loki', ambient as never, 'prompt.submit', {
      session_id: 'rt-socket-closed',
      text: 'work'
    })
    expect(secondaryGateways[0].close).not.toHaveBeenCalled()

    secondaryGateways[0].emitState('closed')
    expect(secondaryGateways[0].close).toHaveBeenCalledOnce()
  })

  it('routes an owner that IS the primary profile onto the primary socket (no active comparison)', async () => {
    const primary = makePrimary()
    setPrimaryGateway(primary as never, 'loki')
    installDesktop()

    const ambient = vi.fn(async (method: string, params?: Record<string, unknown>) => ({
      ambient: true,
      method,
      params
    }))

    // Owner 'loki' equals the PRIMARY profile. There is no active-profile
    // comparison anymore, but gatewayForProfile collapses a primary-profile
    // owner back to the primary socket — so no secondary is spun up. The
    // ambient fn isn't used (routing goes through requestGatewayForProfile),
    // but the request still lands on the one primary gateway.
    const result = await requestForSessionProfile<{ method: string; params: Record<string, unknown> }>(
      'loki',
      ambient as never,
      'session.activate',
      { session_id: 'rt-1' }
    )

    expect(secondaryGateways).toHaveLength(0)
    expect(primary.request).toHaveBeenCalledWith('session.activate', { session_id: 'rt-1' })
    expect(result).toEqual({ method: 'session.activate', params: { session_id: 'rt-1' } })
  })

  it('keeps the ambient dispatcher for sessions with no owning profile', async () => {
    const primary = makePrimary()
    setPrimaryGateway(primary as never, 'default')
    installDesktop()
    await ensureGatewayForProfile('default')

    const ambient = vi.fn(async () => ({ ambient: true }))
    await requestForSessionProfile(null, ambient as never, 'session.usage', { session_id: 'rt-2' })

    expect(ambient).toHaveBeenCalledOnce()
  })
})
