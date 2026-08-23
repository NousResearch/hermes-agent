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
  request: ReturnType<typeof vi.fn>
}> = []

vi.mock('@/hermes', () => ({
  HermesGateway: class {
    connectionState = 'closed'
    connect = vi.fn(async () => {
      this.connectionState = 'open'
    })
    request = vi.fn(async (method: string, params: Record<string, unknown>) => {
      if (this.connectionState !== 'open') {
        throw new Error('gateway is not connected')
      }

      return { method, params }
    })
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
  it('routes ambient when the owner is unknown or already active', () => {
    expect(sessionRpcNeedsProfileRoute(null, 'default')).toBe(false)
    expect(sessionRpcNeedsProfileRoute('', 'default')).toBe(false)
    expect(sessionRpcNeedsProfileRoute('   ', 'loki')).toBe(false)
    expect(sessionRpcNeedsProfileRoute('loki', 'loki')).toBe(false)
    expect(sessionRpcNeedsProfileRoute('default', 'default')).toBe(false)
  })

  it('pins to the owning profile when the active route diverges', () => {
    expect(sessionRpcNeedsProfileRoute('loki', 'default')).toBe(true)
    expect(sessionRpcNeedsProfileRoute('default', 'loki')).toBe(true)
    expect(sessionRpcNeedsProfileRoute('loki', 'hulk')).toBe(true)
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

  it('keeps the ambient dispatcher when the active route already serves the owner', async () => {
    const primary = makePrimary()
    setPrimaryGateway(primary as never, 'default')
    installDesktop()
    await ensureGatewayForProfile('loki')

    const ambient = vi.fn(async (method: string, params?: Record<string, unknown>) => ({
      ambient: true,
      method,
      params
    }))

    const result = await requestForSessionProfile('loki', ambient as never, 'session.activate', { session_id: 'rt-1' })

    expect(ambient).toHaveBeenCalledWith('session.activate', { session_id: 'rt-1' })
    expect(result).toEqual({ ambient: true, method: 'session.activate', params: { session_id: 'rt-1' } })
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
