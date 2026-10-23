import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { backendPath, connection, gatewayWsUrl, parseApiResponse, requestJson, setActiveProfile } from './transport'

beforeEach(() => {
  setActiveProfile(undefined)
  window.__HERMES_BASE_PATH__ = ''
  window.__HERMES_AUTH_REQUIRED__ = false
  window.__HERMES_SESSION_TOKEN__ = 'test-loopback-token'
})
afterEach(() => vi.unstubAllGlobals())

describe('same-origin browser transport', () => {
  it('routes bridge calls to the active profile while preserving explicit request scope', async () => {
    setActiveProfile('coder')
    const fetch = vi.fn().mockResolvedValueOnce(Response.json({})).mockResolvedValueOnce(Response.json({}))
    vi.stubGlobal('fetch', fetch)
    await requestJson({ path: '/api/config' })
    await requestJson({ path: '/api/config', profile: 'reviewer' })
    expect(fetch.mock.calls[0][0]).toBe('/api/config?profile=coder')
    expect(fetch.mock.calls[1][0]).toBe('/api/config?profile=reviewer')
  })

  it('uses the existing backend token on local API calls and keeps explicit profiles', async () => {
    const fetch = vi.fn().mockResolvedValue(Response.json({ ok: true }))
    vi.stubGlobal('fetch', fetch)
    await requestJson({ path: '/api/config', profile: 'coder', method: 'POST', body: { value: 1 } })
    expect(fetch.mock.calls[0][0]).toBe('/api/config?profile=coder')
    const init = fetch.mock.calls[0][1]
    expect(init.credentials).toBe('same-origin')
    expect(init.headers.get('X-Hermes-Session-Token')).toBe('test-loopback-token')
  })

  it('uses cookie auth and mints a new ticket on every gated WebSocket dial', async () => {
    window.__HERMES_AUTH_REQUIRED__ = true

    const fetch = vi.fn().mockResolvedValueOnce(Response.json({ ticket: 'ticket-one' }))
      .mockResolvedValueOnce(Response.json({ ticket: 'ticket-two' }))

    vi.stubGlobal('fetch', fetch)
    const first = await gatewayWsUrl('coder', 'browser')
    const second = await gatewayWsUrl('coder', 'browser')
    expect(first).toMatchObject({ ok: true })
    expect(second).toMatchObject({ ok: true })

    if (!first.ok || !second.ok) {throw new Error('Ticket mint failed')}
    expect(new URL(first.wsUrl).searchParams.get('ticket')).toBe('ticket-one')
    expect(new URL(second.wsUrl).searchParams.get('ticket')).toBe('ticket-two')
    expect(new URL(first.wsUrl).searchParams.get('profile')).toBe('coder')
    expect(new URL(first.wsUrl).searchParams.has('token')).toBe(false)
    expect(fetch.mock.calls[0][1].headers.has('X-Hermes-Session-Token')).toBe(false)
    expect(connection('coder').token).toBe('')
  })

  it('uses loopback bootstrap tokens and supports reverse-proxy prefixes', async () => {
    window.__HERMES_BASE_PATH__ = '/hermes'
    const result = await gatewayWsUrl('coder')

    if (!result.ok) {throw new Error(result.error)}
    expect(new URL(result.wsUrl).pathname).toBe('/hermes/api/ws')
    expect(new URL(result.wsUrl).searchParams.get('token')).toBe('test-loopback-token')
    expect(connection('coder').baseUrl).toBe(window.location.origin + '/hermes')
    expect(backendPath('/api/status')).toBe('/hermes/api/status')
  })

  it('does not send credentials to another backend or absolute API URL', async () => {
    const fetch = vi.fn()
    vi.stubGlobal('fetch', fetch)
    await expect(requestJson({ path: '/api/config', connectionId: 'other' })).rejects.toThrow('own browser tab')
    await expect(requestJson({ path: 'https://other.test/api/config' })).rejects.toThrow('relative')
    await expect(requestJson({ path: '/\\other.test/api/config' })).rejects.toThrow('relative')
    expect(fetch).not.toHaveBeenCalled()
  })

  it('preserves authentication errors and handles non-JSON responses safely', async () => {
    window.__HERMES_AUTH_REQUIRED__ = true
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(Response.json({ detail: 'Unauthorized' }, { status: 401 })))
    expect(await gatewayWsUrl()).toMatchObject({ ok: false, needsOauthLogin: true })
    await expect(parseApiResponse(new Response('<html>login</html>', { status: 401 }))).rejects.toMatchObject({ statusCode: 401, message: '401' })
    await expect(parseApiResponse(new Response('not-json'))).rejects.toThrow('Expected a JSON response')
  })
})


describe('concurrent configuration reads', () => {
  it('shares an in-flight read but refreshes after completion and isolates returned objects', async () => {
    let resolve!: (response: Response) => void

    const fetch = vi.fn().mockImplementationOnce(() => new Promise<Response>(done => { resolve = done }))
      .mockResolvedValueOnce(Response.json({ value: 2 }))

    vi.stubGlobal('fetch', fetch)

    const first = requestJson<{ value: number }>({ path: '/api/config' })
    const second = requestJson<{ value: number }>({ path: '/api/config' })
    expect(fetch).toHaveBeenCalledTimes(1)
    resolve(Response.json({ value: 1 }))
    const [a, b] = await Promise.all([first, second])
    a.value = 9
    expect(b.value).toBe(1)
    expect(await requestJson({ path: '/api/config' })).toEqual({ value: 2 })
    expect(fetch).toHaveBeenCalledTimes(2)
  })

  it('separates profile scopes and never joins a read started before a write', async () => {
    const resolvers: Array<(response: Response) => void> = []
    const fetch = vi.fn().mockImplementation(() => new Promise<Response>(done => { resolvers.push(done) }))
    vi.stubGlobal('fetch', fetch)
    const first = requestJson({ path: '/api/config', profile: 'one' })
    const other = requestJson({ path: '/api/config', profile: 'two' })
    const write = requestJson({ path: '/api/config', profile: 'one', method: 'POST', body: {} })
    const fresh = requestJson({ path: '/api/config', profile: 'one' })
    expect(fetch).toHaveBeenCalledTimes(4)
    resolvers.forEach(resolve => resolve(Response.json({})))
    await Promise.all([first, other, write, fresh])
  })
})


it('refreshes after write settlement instead of joining a read started during the write', async () => {
  const resolvers: Array<(response: Response) => void> = []
  const fetch = vi.fn().mockImplementation(() => new Promise<Response>(done => { resolvers.push(done) }))
  vi.stubGlobal('fetch', fetch)
  const write = requestJson({ path: '/api/config', method: 'POST', body: {} })
  const during = requestJson({ path: '/api/config' })
  resolvers[0](Response.json({}))
  await write
  const after = requestJson({ path: '/api/config' })
  expect(fetch).toHaveBeenCalledTimes(3)
  resolvers[1](Response.json({ version: 'old' }))
  resolvers[2](Response.json({ version: 'new' }))
  expect(await during).toEqual({ version: 'old' })
  expect(await after).toEqual({ version: 'new' })
})

it('evicts failed config reads so a later retry reaches the server', async () => {
  const fetch = vi.fn().mockRejectedValueOnce(new Error('offline')).mockResolvedValueOnce(Response.json({ ok: true }))
  vi.stubGlobal('fetch', fetch)
  await expect(requestJson({ path: '/api/config' })).rejects.toThrow('offline')
  expect(await requestJson({ path: '/api/config' })).toEqual({ ok: true })
  expect(fetch).toHaveBeenCalledTimes(2)
})
