import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { deferred } from '../test/deferred'

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
vi.mock('@/store/session', () => ({ setConnection: vi.fn(), setGatewayState: vi.fn() }))
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
  setPrimaryGateway,
  setPrimaryGatewayConnection
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

  it('background polls on a down local profile leave redialing to the ladder; a user action still dials (#121865)', async () => {
    vi.useFakeTimers()

    try {
      setPrimaryGateway(makePrimary() as never, 'default')

      // The target profile's backend never comes up.
      const getConnection = vi.fn(async (profile: null | string) => {
        if (profile === 'jody') {
          throw new Error('Failed to connect to Hermes gateway')
        }

        return { port: 4242, token: 'primary-token' }
      })

      installDesktop(getConnection)
      await ensureGatewayForProfile('default')

      const jodyDials = () => getConnection.mock.calls.filter(([profile]) => profile === 'jody').length

      // session.control.read against a cross-profile session, polled 20 times in one second.
      for (let index = 0; index < 20; index += 1) {
        await requestGatewayForProfile('jody', 'session.control.read', {}).catch(() => undefined)
        await vi.advanceTimersByTimeAsync(50)
      }

      // Every attempt is two dials into main: the shared-primary probe, then the secondary.
      // Before, all 20 polls attempted (40 dials). Now attempts wait out a window that doubles
      // from 300ms, so one second holds three of them: t=0, t=300, t=900.
      const backgroundDials = jodyDials()
      expect(backgroundDials).toBe(6)

      // Inside the same cooldown, a user action is not held back.
      await requestGatewayForProfile('jody', 'session.activate', {}, undefined, undefined, {
        spawnPriority: 'foreground'
      }).catch(() => undefined)
      expect(jodyDials()).toBeGreaterThan(backgroundDials)
    } finally {
      vi.useRealTimers()
    }
  })

  it('a served request clears dial history, so the next outage starts back at the shortest window (#121865)', async () => {
    vi.useFakeTimers()

    try {
      setPrimaryGateway(makePrimary() as never, 'default')

      let jodyUp = false

      const getConnection = vi.fn(async (profile: null | string) => {
        if (profile === 'jody') {
          if (!jodyUp) {
            throw new Error('Failed to connect to Hermes gateway')
          }

          return { port: 5151, profile, token: 'secondary-token' }
        }

        return { port: 4242, token: 'primary-token' }
      })

      installDesktop(getConnection)
      await ensureGatewayForProfile('default')

      const poll = () => requestGatewayForProfile('jody', 'session.control.read', {}).catch(() => undefined)
      const jodyDials = () => getConnection.mock.calls.filter(([profile]) => profile === 'jody').length

      // Three failed attempts in the first second widen the window to 1200ms.
      for (let index = 0; index < 20; index += 1) {
        await poll()
        await vi.advanceTimersByTimeAsync(50)
      }

      // The backend recovers and serves a request once the widened window has passed.
      jodyUp = true
      await vi.advanceTimersByTimeAsync(2_000)
      await expect(requestGatewayForProfile('jody', 'session.control.read', {})).resolves.toBeDefined()

      // A fresh outage: one failure opens a 300ms window again, so a poll 300ms later dials.
      // Had the old streak survived, the window would be 2400ms and that poll would be held.
      jodyUp = false
      const before = jodyDials()
      await poll()
      await vi.advanceTimersByTimeAsync(300)
      await poll()
      expect(jodyDials() - before).toBe(4)
    } finally {
      vi.useRealTimers()
    }
  })

  it('dials the profile with foreground priority when a Settings-scoped caller asks for it (#111651)', async () => {
    setPrimaryGateway(makePrimary() as never, 'default')

    const getConnection = vi.fn(async (profile: null | string) =>
      profile ? { port: 5151, profile, token: 'secondary-token' } : { port: 4242, token: 'primary-token' }
    )

    installDesktop(getConnection)
    await ensureGatewayForProfile('default')

    await requestGatewayForProfile('worker', 'vault.list', {}, undefined, undefined, { spawnPriority: 'foreground' })

    expect(getConnection).toHaveBeenCalledWith('worker', { priority: 'foreground' })
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
  it('reuses the active primary socket when its registry connection owns the session', async () => {
    const primary = makePrimary()

    const getConnectionFor = vi.fn(async ({ connectionId, profile }) => ({
      connectionId,
      port: 5151,
      profile,
      token: 'secondary-token'
    }))

    setPrimaryGateway(primary as never, 'default')
    setPrimaryGatewayConnection({ connectionId: 'remote-primary' })

    ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = {
      getConnection: vi.fn(),
      getConnectionFor,
      getGatewayWsUrlFor: vi.fn(async () => ({ ok: true as const, wsUrl: 'wss://remote.invalid/api/ws' })),
      touchBackend: vi.fn(async () => undefined)
    }
    await ensureGatewayForProfile('default')

    await openGatewayForAgent('remote-primary', 'default')
    await ensureGatewayForAgent('remote-primary', 'default')

    const result = await requestGatewayForAgent('remote-primary', 'default', 'session.resume', {
      session_id: 'stored-session'
    })

    expect(result).toEqual({
      method: 'session.resume',
      params: { session_id: 'stored-session' }
    })
    expect(primary.request).toHaveBeenCalledWith('session.resume', { session_id: 'stored-session' })
    expect(getConnectionFor).not.toHaveBeenCalled()
    expect(secondaryGateways).toHaveLength(0)
    expect($gateway.get()).toBe(primary)
  })

  it('keeps another profile on the same registry source isolated from the primary', async () => {
    const primary = makePrimary()

    const getConnectionFor = vi.fn(async ({ connectionId, profile }) => ({
      connectionId,
      port: 5151,
      profile,
      token: 'secondary-token'
    }))

    setPrimaryGateway(primary as never, 'default')
    setPrimaryGatewayConnection({ connectionId: 'remote-primary' })

    ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = {
      getConnection: vi.fn(),
      getConnectionFor,
      getGatewayWsUrlFor: vi.fn(async () => ({ ok: true as const, wsUrl: 'wss://remote.invalid/api/ws' })),
      touchBackend: vi.fn(async () => undefined)
    }

    await requestGatewayForAgent('remote-primary', 'research', 'session.resume', {
      session_id: 'research-session'
    })

    expect(getConnectionFor).toHaveBeenCalledWith({ connectionId: 'remote-primary', profile: 'research' })
    expect(primary.request).not.toHaveBeenCalled()
    expect(secondaryGateways).toHaveLength(1)
  })

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

describe('attached shared-remote group turns (#96493)', () => {
  function installAttachedSharedRemote() {
    const getConnectionFor = vi.fn(async ({ connectionId, profile }: { connectionId: string; profile: string }) => ({
      connectionId,
      port: 9119,
      profile,
      sharedRemote: true
    }))

    ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = {
      getConnection: vi.fn(async (profile: null | string) => ({ port: 4242, profile, token: 't' })),
      getConnectionFor,
      getGatewayWsUrlFor: vi.fn(async ({ connectionId, profile }: { connectionId: string; profile: string }) => ({
        ok: true as const,
        wsUrl: `ws://${connectionId}/${profile}`
      })),
      touchBackend: vi.fn(async () => undefined)
    }

    return getConnectionFor
  }

  it('reuses the primary socket for a named profile on the attached shared remote', async () => {
    const primary = makePrimary()
    setPrimaryGateway(primary as never, 'default')
    setPrimaryGatewayConnection({ connectionId: 'homelab' })
    installAttachedSharedRemote()
    await ensureGatewayForProfile('default')

    const release = await retainGatewayForAgent('homelab', 'voter')
    await requestGatewayForAgent('homelab', 'voter', 'session.create', { title: 'Group: room' })
    await requestGatewayForAgent('homelab', 'voter', 'prompt.submit', { session_id: 'rt-1', text: 'hi' })

    expect(secondaryGateways).toHaveLength(0)
    expect(primary.request).toHaveBeenCalledTimes(2)
    expect(primary.request).toHaveBeenNthCalledWith(1, 'session.create', {
      title: 'Group: room',
      profile: 'voter'
    })
    expect(primary.request).toHaveBeenNthCalledWith(2, 'prompt.submit', {
      session_id: 'rt-1',
      text: 'hi',
      profile: 'voter'
    })

    release()
    expect(secondaryGateways).toHaveLength(0)
  })

  it('still dials a secondary when the attached source is not a shared remote', async () => {
    const primary = makePrimary()
    setPrimaryGateway(primary as never, 'default')
    setPrimaryGatewayConnection({ connectionId: 'homelab' })
    ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = {
      getConnection: vi.fn(async (profile: null | string) => ({ port: 4242, profile, token: 't' })),
      getConnectionFor: vi.fn(async ({ connectionId, profile }: { connectionId: string; profile: string }) => ({
        connectionId,
        port: 5151,
        profile,
        sharedRemote: false
      })),
      getGatewayWsUrlFor: vi.fn(async ({ connectionId, profile }: { connectionId: string; profile: string }) => ({
        ok: true as const,
        wsUrl: `ws://${connectionId}/${profile}`
      })),
      touchBackend: vi.fn(async () => undefined)
    }
    await ensureGatewayForProfile('default')

    await requestGatewayForAgent('homelab', 'voter', 'session.create', { title: 'g' })

    expect(secondaryGateways).toHaveLength(1)
    expect(primary.request).not.toHaveBeenCalled()
  })

  it('reuses the primary when the shared-remote probe fails instead of dialing a ghost secondary', async () => {
    const primary = makePrimary()
    setPrimaryGateway(primary as never, 'default')
    setPrimaryGatewayConnection({ connectionId: 'homelab' })
    ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = {
      getConnection: vi.fn(async (profile: null | string) => ({ port: 4242, profile, token: 't' })),
      getConnectionFor: vi.fn(async () => {
        throw new Error('Timed out connecting to profile "voter"')
      }),
      getGatewayWsUrlFor: vi.fn(async () => ({ ok: true as const, wsUrl: 'ws://homelab/voter' })),
      touchBackend: vi.fn(async () => undefined)
    }
    await ensureGatewayForProfile('default')

    await requestGatewayForAgent('homelab', 'voter', 'session.create', { title: 'g' })

    expect(secondaryGateways).toHaveLength(0)
    expect(primary.request).toHaveBeenCalledOnce()
  })

  it('never collapses a pooled LOCAL profile onto the primary when its pool probe fails', async () => {
    // A local Desktop primary is one `hermes serve --profile <primary>` child;
    // pooled profiles get their own child. Sending `session.create` with
    // `profile: sean` to the primary still succeeds (profile_home
    // multiplexing), but the lease then belongs to the primary's pid while
    // every later resume — after a renderer reload or a pool respawn — dials
    // sean's pool backend and is refused with SESSION_NOT_OWNED.
    const primary = makePrimary()
    setPrimaryGateway(primary as never, 'default')
    setPrimaryGatewayConnection({ connectionId: 'local', mode: 'local' })
    ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = {
      getConnection: vi.fn(async (profile: null | string) => ({ mode: 'local', port: 4242, profile, token: 't' })),
      getConnectionFor: vi.fn(async () => {
        throw new Error('Timed out connecting to profile "sean"')
      }),
      getGatewayWsUrlFor: vi.fn(async () => ({ ok: true as const, wsUrl: 'ws://local/sean' })),
      touchBackend: vi.fn(async () => undefined)
    }
    await ensureGatewayForProfile('default')

    await expect(requestGatewayForAgent('local', 'sean', 'session.create', { title: 'g' })).rejects.toThrow(
      /Timed out connecting to profile "sean"/
    )

    expect(primary.request).not.toHaveBeenCalled()
  })

  it('returning to an attached shared remote activates its socket without closing the other source', async () => {
    const primary = makePrimary()
    setPrimaryGateway(primary as never, 'default')
    setPrimaryGatewayConnection({ connectionId: 'homelab' })
    installAttachedSharedRemote()
    await ensureGatewayForProfile('default')

    await openGatewayForAgent('homelab', 'voter')
    expect(await ensureGatewayForAgent('homelab', 'voter')).toBe(true)
    expect(secondaryGateways).toHaveLength(0)

    expect(await ensureGatewayForAgent('local', 'worker')).toBe(true)
    const local = secondaryGateways[0]
    expect($gateway.get()).toBe(local)

    await openGatewayForAgent('homelab', 'voter')
    expect(await ensureGatewayForAgent('homelab', 'voter')).toBe(true)
    expect($gateway.get()).toBe(primary)
    expect(secondaryGateways).toHaveLength(1)
    expect(local.close).not.toHaveBeenCalled()
  })

  it('a delayed shared-remote probe cannot reclaim the foreground from a newer activation', async () => {
    const primary = makePrimary()
    setPrimaryGateway(primary as never, 'default')
    setPrimaryGatewayConnection({ connectionId: 'homelab' })
    const getConnectionFor = installAttachedSharedRemote()
    await ensureGatewayForProfile('default')
    expect(await ensureGatewayForAgent('local', 'worker')).toBe(true)
    const local = $gateway.get()

    const probe = deferred<Awaited<ReturnType<typeof getConnectionFor>>>()
    getConnectionFor.mockReturnValueOnce(probe.promise)
    const staleActivation = ensureGatewayForAgent('homelab', 'voter')
    expect(await ensureGatewayForAgent('local', 'worker')).toBe(true)
    probe.resolve({ connectionId: 'homelab', port: 9119, profile: 'voter', sharedRemote: true })

    expect(await staleActivation).toBe(false)
    expect($gateway.get()).toBe(local)
  })

  it('ensureGatewayForAgent is false when the attached primary socket is closed', async () => {
    const primary = { connectionState: 'closed', request: vi.fn() }
    setPrimaryGateway(primary as never, 'default')
    setPrimaryGatewayConnection({ connectionId: 'homelab' })
    installAttachedSharedRemote()

    expect(await ensureGatewayForAgent('homelab', 'voter')).toBe(false)
    expect(secondaryGateways).toHaveLength(0)
  })
})

describe('session-owner calls for a profile on the shared local host backend (#120005)', () => {
  function installLocalHost(descriptorFor: (profile: string) => Record<string, unknown>) {
    const getConnectionFor = vi.fn(async ({ connectionId, profile }: { connectionId: string; profile: string }) => ({
      connectionId,
      mode: 'local',
      port: 4242,
      token: 't',
      ...descriptorFor(profile)
    }))

    ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = {
      getConnection: vi.fn(async (profile: null | string) => ({ mode: 'local', port: 4242, profile, token: 't' })),
      getConnectionFor,
      getGatewayWsUrlFor: vi.fn(async ({ connectionId, profile }: { connectionId: string; profile: string }) => ({
        ok: true as const,
        wsUrl: `ws://${connectionId}/${profile}`
      })),
      touchBackend: vi.fn(async () => undefined)
    }

    return getConnectionFor
  }

  it('reuses the primary socket when main says the profile rides the host backend (sharedPrimary)', async () => {
    // Under multiplex-only (#118246) one local `hermes serve` serves every
    // profile. A registry secondary here is a second WebSocket to the SAME
    // process: the backend joins it to the chat and the renderer processes
    // every event twice (garbled deltas, duplicate interim bubble).
    const primary = makePrimary()
    setPrimaryGateway(primary as never, 'default')
    setPrimaryGatewayConnection({ connectionId: 'local', mode: 'local' })
    installLocalHost(profile => ({ profile, sharedPrimary: true }))
    await ensureGatewayForProfile('default')

    const release = await retainGatewayForAgent('local', 'work')
    await requestGatewayForAgent('local', 'work', 'session.create', { title: 'g' })
    await requestGatewayForAgent('local', 'work', 'prompt.submit', { session_id: 'rt-1', text: 'hi' })
    release()

    expect(secondaryGateways).toHaveLength(0)
    expect(primary.request).toHaveBeenCalledTimes(2)
    expect(primary.request).toHaveBeenNthCalledWith(1, 'session.create', { title: 'g', profile: 'work' })
    expect(primary.request).toHaveBeenNthCalledWith(2, 'prompt.submit', {
      session_id: 'rt-1',
      text: 'hi',
      profile: 'work'
    })
  })

  it('still dials a secondary for a pooled local profile (isolated backend, #101416)', async () => {
    const primary = makePrimary()
    setPrimaryGateway(primary as never, 'default')
    setPrimaryGatewayConnection({ connectionId: 'local', mode: 'local' })
    installLocalHost(profile => ({ port: 5151, profile }))
    await ensureGatewayForProfile('default')

    await requestGatewayForAgent('local', 'work', 'session.create', { title: 'g' })

    expect(secondaryGateways).toHaveLength(1)
    expect(primary.request).not.toHaveBeenCalled()
  })
})
