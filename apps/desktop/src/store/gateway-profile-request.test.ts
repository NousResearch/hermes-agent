import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

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
vi.mock('@/store/session', () => ({ setGatewayState: vi.fn() }))
vi.mock('@/store/notify-baseline', () => ({ markNativeNotifyBaseline: vi.fn() }))

const {
  $gateway,
  closeSecondaryGateways,
  configureGatewayRegistry,
  ensureGatewayForAgent,
  ensureGatewayForProfile,
  gatewayActivationEpoch,
  disposeSecondariesForConnection,
  openGatewayForAgent,
  pruneSecondaryGateways,
  requestGatewayForAgent,
  requestGatewayForProfile,
  retainGatewayForAgent,
  setPrimaryGateway
} = await import('./gateway')

function installDesktop(getConnection: ReturnType<typeof vi.fn>): void {
  ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = {
    getConnection,
    touchBackend: vi.fn(async () => undefined)
  }
}

function makePrimary() {
  return {
    connectionState: 'open',
    request: vi.fn(async (method: string, params: Record<string, unknown>) => ({ method, params }))
  }
}

beforeEach(async () => {
  secondaryGateways.length = 0
  connectGate = null
  configureGatewayRegistry({ onEvent: vi.fn() })
  closeSecondaryGateways()
})

afterEach(() => {
  closeSecondaryGateways()
  vi.clearAllMocks()
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
})

describe('requestGatewayForProfile', () => {
  it('requests through a pooled profile gateway without changing the active gateway', async () => {
    const primary = makePrimary()
    setPrimaryGateway(primary as never, 'default')
    installDesktop(
      vi.fn(async (profile: null | string) =>
        profile ? { port: 5151, profile, token: 'secondary-token' } : { port: 4242, token: 'primary-token' }
      )
    )
    await ensureGatewayForProfile('default')

    const result = await requestGatewayForProfile('worker', 'profiles.list', { include_sessions: true })

    expect(result).toEqual({ method: 'profiles.list', params: { include_sessions: true } })
    expect(secondaryGateways).toHaveLength(1)
    expect(secondaryGateways[0].request).toHaveBeenCalledWith('profiles.list', { include_sessions: true })
    expect(secondaryGateways[0].close).toHaveBeenCalledOnce()
    expect($gateway.get()).toBe(primary)
  })

  it('uses the primary socket and adds profile scope for a shared global remote route', async () => {
    const primary = makePrimary()
    setPrimaryGateway(primary as never, 'default')
    installDesktop(
      vi.fn(async (profile: null | string) => ({
        port: 4242,
        ...(profile ? { profile, sharedPrimary: true } : {}),
        token: 'primary-token'
      }))
    )
    await ensureGatewayForProfile('default')

    const result = await requestGatewayForProfile('venture', 'session.list', { limit: 20, profile: 'wrong' })

    expect(result).toEqual({ method: 'session.list', params: { limit: 20, profile: 'venture' } })
    expect(primary.request).toHaveBeenCalledWith('session.list', { limit: 20, profile: 'venture' })
    expect(secondaryGateways).toHaveLength(0)
    expect($gateway.get()).toBe(primary)
  })

  it('serializes concurrent requests while a secondary gateway is connecting', async () => {
    let releaseConnect: () => void = () => undefined
    connectGate = new Promise<void>(resolve => {
      releaseConnect = resolve
    })

    const primary = makePrimary()
    setPrimaryGateway(primary as never, 'default')
    installDesktop(
      vi.fn(async (profile: null | string) =>
        profile ? { port: 5151, profile, token: 'secondary-token' } : { port: 4242, token: 'primary-token' }
      )
    )
    await ensureGatewayForProfile('default')

    const first = requestGatewayForProfile('worker', 'profiles.list')
    await vi.waitFor(() => expect(secondaryGateways[0]?.connect).toHaveBeenCalledOnce())
    const second = requestGatewayForProfile('worker', 'profiles.list')
    const guardedSecond = second.catch(error => error)

    await Promise.resolve()
    releaseConnect()

    const [firstResult, secondResult] = await Promise.all([first, guardedSecond])

    expect(firstResult).toEqual({ method: 'profiles.list', params: {} })
    expect(secondResult).toEqual({ method: 'profiles.list', params: {} })
    expect(secondaryGateways[0].connect).toHaveBeenCalledOnce()
    expect(secondaryGateways[0].request).toHaveBeenCalledTimes(2)
    expect(secondaryGateways[0].close).toHaveBeenCalledOnce()
    expect($gateway.get()).toBe(primary)
  })

  it('prevents pruning while a background request is connecting or in flight', async () => {
    let releaseConnect: () => void = () => undefined
    connectGate = new Promise<void>(resolve => {
      releaseConnect = resolve
    })

    const primary = makePrimary()
    setPrimaryGateway(primary as never, 'default')
    installDesktop(
      vi.fn(async (profile: null | string) =>
        profile ? { port: 5151, profile, token: 'secondary-token' } : { port: 4242, token: 'primary-token' }
      )
    )
    await ensureGatewayForProfile('default')

    const request = requestGatewayForProfile('worker', 'profiles.list')
    await vi.waitFor(() => expect(secondaryGateways[0]?.connect).toHaveBeenCalledOnce())

    pruneSecondaryGateways(new Set())
    releaseConnect()

    await expect(request).resolves.toEqual({ method: 'profiles.list', params: {} })
    expect(secondaryGateways[0].close).toHaveBeenCalledOnce()

    pruneSecondaryGateways(new Set())
    expect(secondaryGateways[0].close).toHaveBeenCalledOnce()
  })
})

describe('requestGatewayForAgent', () => {
  it('leases separate registry sockets for duplicate profile names without changing the active gateway', async () => {
    const primary = makePrimary()
    const getConnection = vi.fn(async (profile: null | string) => ({ port: 4242, profile, token: 'legacy-token' }))

    const getConnectionFor = vi.fn(async ({ connectionId, profile }) => ({
      connectionId,
      port: connectionId === 'source-a' ? 5151 : 5252,
      profile,
      token: `${connectionId}-token`
    }))

    const getGatewayWsUrlFor = vi.fn(async ({ connectionId, profile }) => ({
      ok: true as const,
      wsUrl: `ws://${connectionId}/${profile}`
    }))

    setPrimaryGateway(primary as never, 'default')
    ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = {
      getConnection,
      getConnectionFor,
      getGatewayWsUrlFor,
      touchBackend: vi.fn(async () => undefined)
    }
    await ensureGatewayForProfile('default')

    const [fromA, fromB] = await Promise.all([
      requestGatewayForAgent('source-a', 'research', 'session.list', { limit: 1 }),
      requestGatewayForAgent('source-b', 'research', 'session.list', { limit: 2 })
    ])

    expect(fromA).toEqual({ method: 'session.list', params: { limit: 1 } })
    expect(fromB).toEqual({ method: 'session.list', params: { limit: 2 } })
    expect(getConnectionFor).toHaveBeenCalledWith({ connectionId: 'source-a', profile: 'research' })
    expect(getConnectionFor).toHaveBeenCalledWith({ connectionId: 'source-b', profile: 'research' })
    expect(getGatewayWsUrlFor).toHaveBeenCalledWith({ connectionId: 'source-a', profile: 'research' })
    expect(getGatewayWsUrlFor).toHaveBeenCalledWith({ connectionId: 'source-b', profile: 'research' })
    expect(getConnection).not.toHaveBeenCalled()
    expect(secondaryGateways).toHaveLength(2)
    expect(secondaryGateways[0].close).toHaveBeenCalledOnce()
    expect(secondaryGateways[1].close).toHaveBeenCalledOnce()
    expect($gateway.get()).toBe(primary)
  })

  it('routes an explicit local registry descriptor through getConnectionFor', async () => {
    const primary = makePrimary()

    const getConnection = vi.fn(async (profile: null | string) => ({
      mode: 'remote',
      profile,
      wsUrl: 'wss://legacy-remote.invalid/api/ws?token=legacy'
    }))

    const getConnectionFor = vi.fn(async ({ connectionId, profile }) => ({
      connectionId,
      mode: 'local',
      port: 5151,
      profile
    }))

    setPrimaryGateway(primary as never, 'default')
    ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = {
      getConnection,
      getConnectionFor,
      getGatewayWsUrlFor: vi.fn(async ({ connectionId, profile }) => ({
        ok: true as const,
        wsUrl: `ws://${connectionId}/${profile}`
      })),
      touchBackend: vi.fn(async () => undefined)
    }
    await ensureGatewayForProfile('default')

    await expect(requestGatewayForAgent('local', 'worker', 'profiles.list')).resolves.toEqual({
      method: 'profiles.list',
      params: {}
    })
    expect(getConnectionFor).toHaveBeenCalledWith({ connectionId: 'local', profile: 'worker' })
    expect(getConnection).not.toHaveBeenCalled()
    expect($gateway.get()).toBe(primary)
  })

  it('evicts registry sockets when their source is edited or removed', async () => {
    const primary = makePrimary()
    const getConnectionFor = vi.fn(async ({ connectionId, profile }) => ({ connectionId, port: 5151, profile }))
    const onActiveConnectionInvalidated = vi.fn()

    setPrimaryGateway(primary as never, 'default')
    configureGatewayRegistry({ onActiveConnectionInvalidated, onEvent: vi.fn() })
    ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = {
      getConnection: vi.fn(),
      getConnectionFor,
      getGatewayWsUrlFor: vi.fn(async ({ connectionId, profile }) => ({
        ok: true as const,
        wsUrl: `ws://${connectionId}/${profile}`
      })),
      touchBackend: vi.fn(async () => undefined)
    }

    await openGatewayForAgent('source-a', 'research')
    await requestGatewayForAgent('source-a', 'research', 'session.list')
    disposeSecondariesForConnection('source-a')

    expect(secondaryGateways[0].close).toHaveBeenCalledOnce()
    expect(onActiveConnectionInvalidated).not.toHaveBeenCalled()

    await requestGatewayForAgent('source-a', 'research', 'session.list')
    expect(secondaryGateways).toHaveLength(2)
    expect(getConnectionFor).toHaveBeenCalledTimes(2)
  })

  it('invalidates an active pinned source with its window profile and a route epoch', async () => {
    const primary = makePrimary()
    const onActiveConnectionChanged = vi.fn()
    const onActiveConnectionInvalidated = vi.fn()

    setPrimaryGateway(primary as never, 'pinned')
    configureGatewayRegistry({ onActiveConnectionChanged, onActiveConnectionInvalidated, onEvent: vi.fn() })
    ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = {
      getConnection: vi.fn(),
      getConnectionFor: vi.fn(async ({ connectionId, profile }) => ({ connectionId, port: 5151, profile })),
      getGatewayWsUrlFor: vi.fn(async ({ connectionId, profile }) => ({
        ok: true as const,
        wsUrl: `ws://${connectionId}/${profile}`
      })),
      touchBackend: vi.fn(async () => undefined)
    }

    await ensureGatewayForAgent('source-a', 'pinned')
    disposeSecondariesForConnection('source-a')

    const invalidationEpoch = gatewayActivationEpoch()
    expect(onActiveConnectionInvalidated).toHaveBeenCalledWith('pinned', invalidationEpoch)
    expect($gateway.get()).toBe(primary)

    // Same profile, different source advances the guard as soon as activation
    // starts, before its deferred socket connect can finish.
    let releaseConnect: () => void = () => undefined
    connectGate = new Promise<void>(resolve => {
      releaseConnect = resolve
    })
    const sourceBActivation = ensureGatewayForAgent('source-b', 'pinned')
    await vi.waitFor(() => expect(secondaryGateways[1]?.connect).toHaveBeenCalledOnce())
    expect(gatewayActivationEpoch()).toBeGreaterThan(invalidationEpoch)
    releaseConnect()
    await sourceBActivation
    expect(onActiveConnectionChanged).toHaveBeenLastCalledWith(expect.objectContaining({ connectionId: 'source-b' }))
  })

  it('does not activate or publish a source invalidated while its dial is pending', async () => {
    const primary = makePrimary()
    const onActiveConnectionChanged = vi.fn()

    setPrimaryGateway(primary as never, 'default')
    configureGatewayRegistry({ onActiveConnectionChanged, onEvent: vi.fn() })
    ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = {
      getConnection: vi.fn(async profile => ({ port: 4242, profile })),
      getConnectionFor: vi.fn(async ({ connectionId, profile }) => ({ connectionId, port: 5151, profile })),
      getGatewayWsUrlFor: vi.fn(async ({ connectionId, profile }) => ({
        ok: true as const,
        wsUrl: `ws://${connectionId}/${profile}`
      })),
      touchBackend: vi.fn(async () => undefined)
    }
    await ensureGatewayForProfile('default')

    let releaseConnect: () => void = () => undefined
    connectGate = new Promise<void>(resolve => {
      releaseConnect = resolve
    })
    const pendingActivation = ensureGatewayForAgent('source-b', 'default')
    await vi.waitFor(() => expect(secondaryGateways[0]?.connect).toHaveBeenCalledOnce())

    disposeSecondariesForConnection('source-b')
    releaseConnect()
    await pendingActivation

    expect(secondaryGateways[0].close).toHaveBeenCalled()
    expect(onActiveConnectionChanged).not.toHaveBeenCalled()
    expect($gateway.get()).toBe(primary)
  })

  it('does not activate or publish a source whose activation owner aborted while dialing', async () => {
    const primary = makePrimary()
    const onActiveConnectionChanged = vi.fn()
    const controller = new AbortController()

    setPrimaryGateway(primary as never, 'default')
    configureGatewayRegistry({ onActiveConnectionChanged, onEvent: vi.fn() })
    ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = {
      getConnection: vi.fn(async profile => ({ port: 4242, profile })),
      getConnectionFor: vi.fn(async ({ connectionId, profile }) => ({ connectionId, port: 5151, profile })),
      getGatewayWsUrlFor: vi.fn(async ({ connectionId, profile }) => ({
        ok: true as const,
        wsUrl: `ws://${connectionId}/${profile}`
      })),
      touchBackend: vi.fn(async () => undefined)
    }
    await ensureGatewayForProfile('default')

    let releaseConnect: () => void = () => undefined
    connectGate = new Promise<void>(resolve => {
      releaseConnect = resolve
    })
    const abandoned = ensureGatewayForAgent('source-b', 'default', { signal: controller.signal })
    await vi.waitFor(() => expect(secondaryGateways[0]?.connect).toHaveBeenCalledOnce())

    controller.abort()
    releaseConnect()

    expect(await abandoned).toBe(false)
    expect(onActiveConnectionChanged).not.toHaveBeenCalled()
    expect($gateway.get()).toBe(primary)
  })
})

describe('retainGatewayForAgent (#93602)', () => {
  function installRegistryDesktop() {
    ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = {
      getConnection: vi.fn(async (profile: null | string) => ({ port: 4242, profile, token: 't' })),
      getConnectionFor: vi.fn(async ({ connectionId, profile }) => ({ connectionId, port: 5151, profile })),
      getGatewayWsUrlFor: vi.fn(async ({ connectionId, profile }) => ({
        ok: true as const,
        wsUrl: `ws://${connectionId}/${profile}`
      })),
      touchBackend: vi.fn(async () => undefined)
    }
  }

  it('holds the registry socket across multiple leased requests — no mid-turn disposal', async () => {
    const primary = makePrimary()
    setPrimaryGateway(primary as never, 'default')
    installRegistryDesktop()
    await ensureGatewayForProfile('default')

    const release = await retainGatewayForAgent('mini', 'helper')

    // The whole member-turn RPC sequence: each call takes and releases its own
    // per-request lease. Without the retain, refcount 0 between calls disposes
    // the socket and the runtime session with it.
    await requestGatewayForAgent('mini', 'helper', 'session.create', { title: 'g' })
    await requestGatewayForAgent('mini', 'helper', 'image.attach_bytes', { session_id: 'rt-1' })
    await requestGatewayForAgent('mini', 'helper', 'prompt.submit', { session_id: 'rt-1', text: 'hi' })

    expect(secondaryGateways).toHaveLength(1)
    expect(secondaryGateways[0].close).not.toHaveBeenCalled()
    expect(secondaryGateways[0].request).toHaveBeenCalledTimes(3)

    release()
    expect(secondaryGateways[0].close).toHaveBeenCalledOnce()
  })

  it('release is idempotent — a double release never underflows into a foreign disposal', async () => {
    const primary = makePrimary()
    setPrimaryGateway(primary as never, 'default')
    installRegistryDesktop()
    await ensureGatewayForProfile('default')

    const release = await retainGatewayForAgent('mini', 'helper')
    release()
    release()

    expect(secondaryGateways[0].close).toHaveBeenCalledOnce()

    // A fresh retain still works after the double release.
    const again = await retainGatewayForAgent('mini', 'helper')
    await requestGatewayForAgent('mini', 'helper', 'prompt.submit', { session_id: 'rt-2', text: 'hi' })
    expect(secondaryGateways[1].close).not.toHaveBeenCalled()
    again()
    expect(secondaryGateways[1].close).toHaveBeenCalledOnce()
  })

  it('without the retain, the leased socket closes after each request (the #93602 race)', async () => {
    const primary = makePrimary()
    setPrimaryGateway(primary as never, 'default')
    installRegistryDesktop()
    await ensureGatewayForProfile('default')

    await requestGatewayForAgent('mini', 'helper', 'session.create', { title: 'g' })

    // Refcount hit 0 → disposed: this is the socket close that reaps the
    // runtime session server-side and makes the later prompt.submit 4001.
    expect(secondaryGateways[0].close).toHaveBeenCalledOnce()
  })

  it('plain-profile retain leases the pooled profile socket and releases it', async () => {
    const primary = makePrimary()
    setPrimaryGateway(primary as never, 'default')
    installDesktop(
      vi.fn(async (profile: null | string) =>
        profile ? { port: 5151, profile, token: 'secondary-token' } : { port: 4242, token: 'primary-token' }
      )
    )
    await ensureGatewayForProfile('default')

    const release = await retainGatewayForAgent(null, 'worker')
    await requestGatewayForProfile('worker', 'session.create', { title: 'g' })
    await requestGatewayForProfile('worker', 'prompt.submit', { session_id: 'rt-1', text: 'hi' })

    expect(secondaryGateways).toHaveLength(1)
    expect(secondaryGateways[0].close).not.toHaveBeenCalled()

    release()
    expect(secondaryGateways[0].close).toHaveBeenCalledOnce()
  })
})
