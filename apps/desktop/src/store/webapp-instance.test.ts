import { waitFor } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { createWebappInstanceSession } from './webapp-instance'

const identity = { organizationId: 'org_1', sessionId: 'sess_1', userId: 'user_1' }
const ownHost = 'atouch-agent-web.unitalk.ai'
const otherHost = 'other-agent-web.unitalk.ai'
const instance = (webDomain = ownHost, ownerId = 'org_1') => ({
  _id: 'instance_1', app: 'hermes', ownerId, status: 'ready', userId: 'user_1', webDomain
})
const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), {
  headers: { 'Content-Type': 'application/json' }, status
})

function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>(done => { resolve = done })

  return { promise, resolve }
}

function setup(host = ownHost) {
  const calls: string[] = []
  const fetcher = vi.fn<typeof fetch>(async input => {
    const path = new URL(String(input)).pathname
    calls.push(path)
    switch (path) {
      case '/api/instance': return json({ instance: instance(host) })
      case '/api/auth-init': return json({ url: `https://${host}/` })
      case '/hermes/auth/logout': return json({ ok: true })
      case '/hermes/api/auth/me': return json({ user_id: 'basic-user' })
      default: throw new Error(`Unexpected request: ${path}`)
    }
  })
  const navigate = vi.fn()
  const pause = vi.fn()
  const refreshSession = vi.fn(async () => undefined)
  const session = createWebappInstanceSession({
    basePath: '/hermes', fetch: fetcher, navigate, origin: `https://${ownHost}`, pause, refreshSession
  })

  return { calls, fetcher, navigate, pause, refreshSession, session }
}

describe('Webapp instance session', () => {
  it('waits for the Clerk change and cookie refresh, then logs out and initializes the destination with cookies only', async () => {
    const { calls, fetcher, navigate, refreshSession, session } = setup(otherHost)
    const changed = deferred<void>()
    const synchronized = session.synchronize(identity, () => changed.promise)
    expect(session.state.get().status).toBe('checking')
    expect(fetcher).not.toHaveBeenCalled()
    changed.resolve()
    await synchronized

    expect(refreshSession).toHaveBeenCalledOnce()
    expect(calls).toEqual(['/api/instance', '/hermes/auth/logout', '/api/auth-init'])
    expect(navigate).toHaveBeenCalledWith(`https://${otherHost}/`)

    const unitalkCalls = fetcher.mock.calls.filter(([input]) =>
      ['/api/instance', '/api/auth-init'].includes(new URL(String(input)).pathname)
    )

    expect(unitalkCalls).toHaveLength(2)

    for (const [input, init] of unitalkCalls) {
      expect(new URL(String(input)).origin).toBe('https://unitalk.ai')
      expect(init?.credentials).toBe('include')
      expect(init?.cache).toBe('no-store')
      expect(new Headers(init?.headers).get('Accept')).toBe('application/json')
      expect(new Headers(init?.headers).has('Authorization')).toBe(false)

      if (new URL(String(input)).pathname === '/api/auth-init') {
        expect(new URL(String(input)).searchParams.get('session')).toBe('sess_1')
        expect(new URL(String(input)).searchParams.get('instanceId')).toBe('instance_1')
      }
    }
  })

  it('keeps a valid same-host session, logs out when Clerk signs out, and bootstraps it again after login', async () => {
    const { calls, navigate, session } = setup()
    await session.synchronize(identity)
    expect(session.state.get().status).toBe('ready')
    expect(calls).toEqual(['/api/instance', '/hermes/api/auth/me'])
    await session.synchronize(null)
    expect(session.state.get().status).toBe('signed-out')
    await session.synchronize(identity)
    expect(session.state.get().status).toBe('ready')
    expect(calls).toEqual([
      '/api/instance', '/hermes/api/auth/me', '/hermes/auth/logout', '/api/instance', '/api/auth-init'
    ])
    expect(navigate).not.toHaveBeenCalled()
  })

  it.each([
    { response: null, status: 'missing' },
    { response: { ...instance(), status: 'stopped' }, status: 'inactive' },
    { response: { ...instance(), status: 'pending', webDomain: '' }, status: 'inactive' }
  ])('ends the old session and blocks access when the instance is $status', async ({ response, status }) => {
    const { calls, fetcher, navigate, session } = setup()
    fetcher.mockResolvedValueOnce(json({ instance: response }))
    await session.synchronize(identity)
    expect(session.state.get().status).toBe(status)
    expect(calls).toEqual(['/hermes/auth/logout'])
    expect(navigate).not.toHaveBeenCalled()
  })

  it.each([
    { httpStatus: 401, message: 'Unauthorized', state: 'unauthorized', logout: true },
    { httpStatus: 403, message: 'Active organization required', state: 'organization-required', logout: true },
    { httpStatus: 404, message: 'Instance not found', state: 'missing', logout: true },
    { httpStatus: 503, message: 'Instance could not be retrieved', state: 'error', logout: false }
  ])('handles the instance endpoint HTTP $httpStatus contract and can recover on retry', async ({
    httpStatus, message, state, logout
  }) => {
    const { fetcher, navigate, session } = setup()
    fetcher.mockResolvedValueOnce(json({ error: message }, httpStatus))
    await session.synchronize(identity)
    expect(session.state.get().status).toBe(state)
    expect(session.state.get().error).toBe(message)
    expect(fetcher.mock.calls.map(([url]) => new URL(String(url)).pathname)).toEqual(
      logout ? ['/api/instance', '/hermes/auth/logout'] : ['/api/instance']
    )
    expect(navigate).not.toHaveBeenCalled()
    await session.retry()
    expect(session.state.get().status).toBe('ready')
  })

  it('treats a plain-text auth-init 401 as expired authentication, not as a missing instance', async () => {
    const { fetcher, navigate, session } = setup(otherHost)
    const normalFetch = fetcher.getMockImplementation()!
    fetcher.mockImplementation((input, init) => new URL(String(input)).pathname === '/api/auth-init'
      ? Promise.resolve(new Response('Unauthorized', { status: 401 }))
      : normalFetch(input, init))
    await session.synchronize(identity)
    expect(session.state.get().status).toBe('unauthorized')
    expect(navigate).not.toHaveBeenCalled()
  })

  it('ignores a late lookup for the previous organization, even when the transport ignores abort', async () => {
    const { calls, fetcher, navigate, session } = setup()
    const first = deferred<Response>()
    fetcher.mockImplementationOnce(() => first.promise)
    const old = session.synchronize(identity)
    await waitFor(() => expect(fetcher).toHaveBeenCalledOnce())
    const next = session.synchronize({ ...identity, organizationId: 'org_2' })
    fetcher.mockResolvedValueOnce(json({ instance: instance(ownHost, 'org_2') }))
    first.resolve(json({ instance: instance(otherHost) }))
    await Promise.all([old, next])
    expect(session.state.get().status).toBe('ready')
    expect(session.state.get().identity?.organizationId).toBe('org_2')
    expect(calls).toEqual(['/hermes/api/auth/me'])
    expect(navigate).not.toHaveBeenCalled()
  })

  it('finishes an in-flight same-host cookie write before a superseding logout', async () => {
    const { calls, fetcher, navigate, session } = setup()
    await session.synchronize(null)
    const initialized = deferred<Response>()
    const normalFetch = fetcher.getMockImplementation()!
    fetcher.mockImplementation(async (input, init) => {
      if (new URL(String(input)).pathname === '/api/auth-init') {
        calls.push('/api/auth-init')
        return initialized.promise
      }
      return normalFetch(input, init)
    })
    const login = session.synchronize(identity)
    await waitFor(() => expect(calls).toContain('/api/auth-init'))
    const logout = session.synchronize(null)
    initialized.resolve(json({ url: `https://${ownHost}/` }))
    await Promise.all([login, logout])
    expect(calls).toEqual(['/hermes/auth/logout', '/api/instance', '/api/auth-init', '/hermes/auth/logout'])
    expect(session.state.get().status).toBe('signed-out')
    expect(navigate).not.toHaveBeenCalled()
  })

  it('does not let an intermediate Clerk org update cancel a newer queued org choice', async () => {
    const { fetcher, navigate, session } = setup()
    const changed = deferred<void>()
    const changeFirst = vi.fn(() => changed.promise)
    const first = session.synchronize(identity, changeFirst)
    await waitFor(() => expect(changeFirst).toHaveBeenCalledOnce())
    const secondIdentity = { ...identity, organizationId: 'org_2' }
    const changeSecond = vi.fn(async () => undefined)
    const second = session.synchronize(secondIdentity, changeSecond)
    // Clerk finishes the first setActive after the second action has been queued.
    const observed = session.synchronize(identity)
    fetcher.mockResolvedValueOnce(json({ instance: instance(ownHost, 'org_2') }))
    changed.resolve()
    await Promise.all([first, second, observed])
    expect(changeSecond).toHaveBeenCalledOnce()
    expect(session.state.get().identity).toEqual(secondIdentity)
    expect(session.state.get().status).toBe('ready')
    expect(navigate).not.toHaveBeenCalled()
  })

  it.each([
    { response: new Response('Instance not found or access denied', { status: 200 }), message: 'Instance not found or access denied' },
    { response: json({ error: 'Instance not found or access denied' }, 403), message: 'Instance not found or access denied' },
    { response: json({ error: 'Hermes web domain not found' }, 404), message: 'Hermes web domain not found' },
    { response: json({ error: 'Activate your AI Collaborator before opening it' }, 409), message: 'Activate your AI Collaborator before opening it' },
    { response: json({ url: 'https://wrong-agent-web.unitalk.ai/' }), message: 'different instance URL' }
  ])('surfaces auth-init errors without navigating: $message', async ({ response, message }) => {
    const { fetcher, navigate, session } = setup(otherHost)
    const normalFetch = fetcher.getMockImplementation()!
    fetcher.mockImplementation((input, init) => new URL(String(input)).pathname === '/api/auth-init'
      ? Promise.resolve(response)
      : normalFetch(input, init))
    await session.synchronize(identity)
    expect(session.state.get().status).toBe('error')
    expect(session.state.get().error).toContain(message)
    expect(navigate).not.toHaveBeenCalled()
  })

  it('does not use an instance returned for stale organization cookies', async () => {
    const { calls, navigate, session } = setup()
    await session.synchronize({ ...identity, organizationId: 'org_other' })
    expect(session.state.get().status).toBe('error')
    expect(calls).toEqual(['/api/instance'])
    expect(navigate).not.toHaveBeenCalled()
  })

  it('accepts a new Clerk organization observation after a failed account-panel change', async () => {
    const { fetcher, session } = setup()
    await expect(session.synchronize(identity, async () => { throw new Error('Clerk unavailable') }))
      .rejects.toThrow('Clerk unavailable')
    fetcher.mockResolvedValueOnce(json({ instance: instance(ownHost, 'org_2') }))
    await session.synchronize({ ...identity, organizationId: 'org_2' })
    expect(session.state.get().status).toBe('ready')
  })

  it('can retry a failed lookup and does not repeat a completed reconciliation on rerender', async () => {
    const { fetcher, session } = setup()
    fetcher.mockRejectedValueOnce(new TypeError('Failed to fetch'))
    await session.synchronize(identity)
    expect(session.state.get().status).toBe('error')
    await session.retry()
    expect(session.state.get().status).toBe('ready')
    const count = fetcher.mock.calls.length
    await session.synchronize({ ...identity })
    expect(fetcher.mock.calls).toHaveLength(count)
  })
})
