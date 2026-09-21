import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { host, type PluginProfileRoute } from '@/sdk'

// Actual SDK descriptor overload -> actual registry. Only native descriptors
// and decoded socket responders are synthetic; no Files consumer or Save path.
const sockets = vi.hoisted(() => ({ created: vi.fn() }))
vi.mock('@/hermes', async importActual => ({
  ...(await importActual<Record<string, unknown>>()),
  HermesGateway: class {
    constructor() {
      sockets.created(this)
    }

    connectionState = 'closed'
    onEvent = () => () => undefined
    onState = () => () => undefined
    close = vi.fn()
    connect = vi.fn(async () => {
      this.connectionState = 'open'
    })

    request = vi.fn(async (method: string, params: Record<string, unknown>) => {
      expect(this.connectionState).toBe('open')

      return { method, params }
    })
  }
}))

const {
  $gateway, closeSecondaryGateways, configureGatewayRegistry, disposeSecondariesForConnection, ensureGatewayForAgent,
  openGatewayForAgent, requestGatewayForAgent, retainGatewayForAgent,
  setPrimaryGateway, setPrimaryGatewayConnectionId

} = await import('./gateway')

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (error: Error) => void
  const promise = new Promise<T>((yes, no) => {
    resolve = yes
    reject = no
  })

  return { promise, resolve, reject }
}

function primary() {
  return {
    connectionState: 'open',
    request: vi.fn(async (method: string, params: Record<string, unknown>) => ({ method, params }))
  }
}

function route(connectionId = 'gateway-a', profile = 'reviewer'): PluginProfileRoute {
  return { connectionId, mode: 'remote', profile, targetProfile: profile }
}

function desktop(getConnectionFor = vi.fn(async () => ({ sharedRemote: true }))) {
  const api = {
    getConnectionFor,
    getConnection: vi.fn(async (profile: string) => ({ port: 5151, profile })),
    getGatewayWsUrlFor: vi.fn(async () => ({ ok: true, wsUrl: 'wss://synthetic.invalid/api/ws' }))
  }

  window.hermesDesktop = api as unknown as typeof window.hermesDesktop

  return api
}

const originalDesktop = window.hermesDesktop

beforeEach(() => {
  configureGatewayRegistry({ onEvent: vi.fn() })
  closeSecondaryGateways()
  setPrimaryGateway(null)
  sockets.created.mockClear()
})

afterEach(() => {
  closeSecondaryGateways()
  setPrimaryGateway(null)
  vi.restoreAllMocks()
  window.hermesDesktop = originalDesktop
})

const ownerChanges = ['gateway', 'profile', 'connection', 'gateway-aba', 'profile-aba', 'connection-aba'] as const
const outcomes = ['success', 'failure', 'isolated'] as const
const races = ownerChanges.flatMap(change => outcomes.map(outcome => ({ change, outcome })))

it('leases the SDK shared route across RPCs and rejects an in-flight primary ABA', async () => {
  desktop()

  const a = primary()
  setPrimaryGateway(a as never)
  setPrimaryGatewayConnectionId('gateway-a')

  const requestedRoute = route()
  const acquiring = host.acquireProfileRoute(requestedRoute)
  Object.assign(requestedRoute, {
    connectionId: 'gateway-b', profile: 'other-profile', targetProfile: 'other-profile'
  })
  const lease = await acquiring
  expect(lease.route).toEqual(route())
  const held = deferred<{ method: string; params: Record<string, unknown> }>()
  a.request.mockReturnValueOnce(held.promise)
  const response = lease.request('groups.import_history', { private_history: 'bytes' })
  await vi.waitFor(() => expect(a.request).toHaveBeenCalledOnce())

  setPrimaryGatewayConnectionId('gateway-b')
  setPrimaryGatewayConnectionId('gateway-a')
  held.resolve({ method: 'groups.import_history', params: { private_history: 'bytes' } })

  await expect(response).rejects.toThrow('route lease expired')
  expect(() => lease.assertCurrent()).toThrow('route lease expired')
  lease.release()

  const fresh = await host.acquireProfileRoute(route())
  await expect(fresh.request('groups.capabilities')).resolves.toEqual({
    method: 'groups.capabilities',
    params: { profile: 'reviewer' }
  })
  fresh.release()
})

it('keeps an isolated lease on one socket and fences edit, removal, and same-id recovery', async () => {
  desktop(vi.fn(async () => ({ sharedRemote: false })))
  const a = primary()
  setPrimaryGateway(a as never)
  setPrimaryGatewayConnectionId('gateway-a')

  const lease = await host.acquireProfileRoute(route('source-a'))
  const firstSocket = sockets.created.mock.calls[0][0]
  const held = deferred<{ ok: boolean }>()
  firstSocket.request.mockReturnValueOnce(held.promise)
  const response = lease.request('groups.import_history', { private_history: 'bytes' })
  await vi.waitFor(() => expect(firstSocket.request).toHaveBeenCalledOnce())

  disposeSecondariesForConnection('source-a', { redial: true })
  disposeSecondariesForConnection('source-a', { redial: true })
  held.resolve({ ok: true })

  await expect(response).rejects.toThrow('route lease expired')
  expect(firstSocket.close).not.toHaveBeenCalled()
  lease.release()
  await vi.waitFor(() => expect(sockets.created).toHaveBeenCalledTimes(2))
  expect(firstSocket.close).toHaveBeenCalledOnce()

  const fresh = await host.acquireProfileRoute(route('source-a'))
  await expect(fresh.request('groups.capabilities')).resolves.toEqual({
    method: 'groups.capabilities',
    params: {}
  })
  const freshSocket = sockets.created.mock.calls[1][0]
  const callsBeforeRemoval = freshSocket.request.mock.calls.length

  disposeSecondariesForConnection('source-a')

  expect(() => fresh.assertCurrent()).toThrow('route lease expired')
  await expect(fresh.request('groups.import_history', { private_history: 'later-bytes' })).rejects.toThrow(
    'route lease expired'
  )
  expect(freshSocket.request).toHaveBeenCalledTimes(callsBeforeRemoval)
  expect(freshSocket.close).toHaveBeenCalledOnce()
  fresh.release()

  const recovered = await host.acquireProfileRoute(route('source-a'))
  await expect(recovered.request('groups.capabilities')).resolves.toEqual({
    method: 'groups.capabilities',
    params: {}
  })
  recovered.release()

})

it.each(races)(
  'rejects private SDK dispatch and sibling consumers after $change with a held $outcome descriptor',
  async ({ change, outcome }) => {
    const held = deferred<unknown>()
    const descriptor = vi.fn().mockReturnValue(held.promise)
    const api = desktop(descriptor)
    const a = primary()
    const b = primary()
    setPrimaryGateway(a as never)
    setPrimaryGatewayConnectionId('gateway-a')
    await ensureGatewayForAgent('gateway-a', 'default')
    const payload = { query: 'private-a', session_id: 'session-a', profile: 'must-be-replaced' }

    // Both SDK overload calls enter the real descriptor await before any writer
    // changes ownership. Sibling consumers must not fall through to a secondary.
    const pending = [
      host.requestProfile(route(), 'session.list', payload).catch(error => error),
      host.requestProfile(route(), 'session.resume', payload, 12_345).catch(error => error),
      openGatewayForAgent('gateway-a', 'reviewer').catch(error => error),
      ensureGatewayForAgent('gateway-a', 'reviewer').catch(error => error),
      retainGatewayForAgent('gateway-a', 'reviewer').catch(error => error)
    ]

    expect(descriptor).toHaveBeenCalledTimes(pending.length)
    expect(descriptor.mock.calls.every(([arg]) => arg.connectionId === 'gateway-a' && arg.profile === 'reviewer')).toBe(
      true
    )
    expect(a.request).not.toHaveBeenCalled()
    expect(sockets.created).not.toHaveBeenCalled()

    if (change.startsWith('gateway')) {
      setPrimaryGateway(b as never)
      setPrimaryGatewayConnectionId('gateway-b')

      if (change.endsWith('aba')) {
        setPrimaryGateway(a as never)
        setPrimaryGatewayConnectionId('gateway-a')
      }
    } else if (change.startsWith('profile')) {
      setPrimaryGateway(a as never, 'other')

      if (change.endsWith('aba')) {
        setPrimaryGateway(a as never, 'default')
      }
    } else {
      setPrimaryGatewayConnectionId('gateway-b')

      if (change.endsWith('aba')) {
        setPrimaryGatewayConnectionId('gateway-a')
      }
    }

    if (outcome === 'failure') {
      held.reject(new Error('native descriptor failed'))
    } else {
      held.resolve({ sharedRemote: outcome === 'success', port: 5151, profile: 'reviewer' })
    }

    const results = await Promise.all(pending)
    expect(a.request).not.toHaveBeenCalled()
    expect(b.request).not.toHaveBeenCalled()
    expect(sockets.created).not.toHaveBeenCalled()
    expect(api.getGatewayWsUrlFor).not.toHaveBeenCalled()
    expect(api.getConnection).not.toHaveBeenCalled()
    results.forEach(result => expect(result).toEqual(new Error('Hermes gateway connection owner changed')))

    // A current, correctly qualified request still works after every rejection,
    // including ABA back to the exact original gateway/profile/connection tuple.
    const freshId = change === 'gateway' || change === 'connection' ? 'gateway-b' : 'gateway-a'
    const current = change === 'gateway' ? b : a
    descriptor.mockResolvedValue({ sharedRemote: true })
    const freshParams = { session_id: 'fresh-session', profile: 'wrong' }
    await expect(
      host.requestProfile(route(freshId, 'fresh-worker'), 'session.resume', freshParams, 23_456)
    ).resolves.toEqual({ method: 'session.resume', params: { ...freshParams, profile: 'fresh-worker' } })
    expect(descriptor).toHaveBeenLastCalledWith({ connectionId: freshId, profile: 'fresh-worker' })
    expect(current.request).toHaveBeenCalledExactlyOnceWith(
      'session.resume',
      { ...freshParams, profile: 'fresh-worker' },
      23_456,
      undefined
    )
    expect(sockets.created).not.toHaveBeenCalled()
    expect(payload.profile).toBe('must-be-replaced')
  }
)

it.each(outcomes)('honors supplied request cancellation across a held %s descriptor', async outcome => {
  const held = deferred<unknown>()
  const descriptor = vi.fn().mockReturnValue(held.promise)
  const api = desktop(descriptor)
  const a = primary()
  const controller = new AbortController()
  setPrimaryGateway(a as never)
  setPrimaryGatewayConnectionId('gateway-a')
  await ensureGatewayForAgent('gateway-a', 'default')

  // The SDK has no signal parameter; exercise the actual registry's optional
  // signal contract, without inventing a cancellation overload for the SDK.
  const pending = requestGatewayForAgent(
    'gateway-a',
    'reviewer',
    'session.list',
    { query: 'cancelled-private' },
    undefined,
    controller.signal
  ).catch(error => error)

  expect(descriptor).toHaveBeenCalledOnce()
  controller.abort()

  if (outcome === 'failure') {
    held.reject(new Error('native descriptor failed'))
  } else {
    held.resolve({ sharedRemote: outcome === 'success' })
  }

  expect(await pending).toBe(controller.signal.reason)
  expect(a.request).not.toHaveBeenCalled()
  expect(sockets.created).not.toHaveBeenCalled()
  expect(api.getGatewayWsUrlFor).not.toHaveBeenCalled()
  descriptor.mockResolvedValue({ sharedRemote: true })
  const live = new AbortController()
  await requestGatewayForAgent('gateway-a', 'reviewer', 'session.list', { query: 'fresh' }, undefined, live.signal)
  expect(a.request).toHaveBeenCalledExactlyOnceWith(
    'session.list',
    { query: 'fresh', profile: 'reviewer' },
    undefined,
    live.signal
  )
})

it.each(['shared', 'probe-error', 'isolated', 'exact-primary', 'plain-profile'] as const)(
  'preserves current-owner %s routing and explicit request arguments',
  async kind => {
    const descriptor = vi.fn(async () => {
      if (kind === 'probe-error') {
        throw new Error('native descriptor failed')
      }

      return { sharedRemote: kind !== 'isolated', port: 5151, profile: 'reviewer' }
    })

    const api = desktop(descriptor)
    const a = primary()
    setPrimaryGateway(a as never)
    setPrimaryGatewayConnectionId('gateway-a')
    await ensureGatewayForAgent('gateway-a', 'default')
    // No-op primary writers do not retire valid pending probes.
    const params = { session_id: 'current-session', profile: 'caller-profile' }
    const target =
      kind === 'plain-profile' ? 'reviewer' : route('gateway-a', kind === 'exact-primary' ? 'default' : 'reviewer')
    const pending = host.requestProfile(target, 'session.resume', params)
    setPrimaryGateway(a as never)
    setPrimaryGatewayConnectionId('gateway-a')
    const result = await pending

    if (kind === 'isolated' || kind === 'plain-profile') {
      expect(a.request).not.toHaveBeenCalled()
      expect(sockets.created).toHaveBeenCalledOnce()
      const secondary = sockets.created.mock.calls[0][0]
      expect(secondary.request).toHaveBeenCalledExactlyOnceWith('session.resume', params)
      expect(result).toEqual({ method: 'session.resume', params })
      expect(secondary.close).toHaveBeenCalledOnce()
    } else {
      const scoped = kind === 'exact-primary' ? params : { ...params, profile: 'reviewer' }
      expect(a.request).toHaveBeenCalledExactlyOnceWith('session.resume', scoped)
      expect(result).toEqual({ method: 'session.resume', params: scoped })
      expect(sockets.created).not.toHaveBeenCalled()
    }

    if (kind === 'exact-primary' || kind === 'plain-profile') {
      expect(descriptor).not.toHaveBeenCalled()
    } else {
      expect(descriptor).toHaveBeenCalledWith({ connectionId: 'gateway-a', profile: 'reviewer' })
    }

    if (kind === 'plain-profile') {
      expect(api.getConnection).toHaveBeenCalledWith('reviewer')
    } else {
      expect(api.getConnection).not.toHaveBeenCalled()
    }

    expect($gateway.get()).toBe(a)
    expect(params.profile).toBe('caller-profile')
  }
)
