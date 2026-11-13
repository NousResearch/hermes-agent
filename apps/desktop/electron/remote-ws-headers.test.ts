import { describe, expect, it, vi } from 'vitest'

import {
  applyRemoteRequestHeaders,
  attachRemoteRequestHeaderListener,
  collectRemoteHeaderSources,
  createRegistryGatewayWsUrlHandler,
  createRemoteWsHeaderStore,
  oauthLoginLoadUrlOptions,
  type RegistryGatewayWsConnection,
  resolveRemoteRequestHeaders
} from './remote-ws-headers'

const accessHeaders = {
  'CF-Access-Client-Id': 'client-id',
  'CF-Access-Client-Secret': 'client-secret'
}

function createHarness(connection: RegistryGatewayWsConnection) {
  const store = createRemoteWsHeaderStore()
  const ensureBackend = vi.fn(async () => connection)
  const mintTicket = vi.fn(async () => 'fresh-ticket')

  const handler = createRegistryGatewayWsUrlHandler({
    ensureBackend,
    mintTicket,
    buildTicketUrl: baseUrl => `${baseUrl.replace(/^https:/, 'wss:')}/api/ws?region=us&ticket=fresh-ticket&profile=old`,
    rememberHeaders: store.remember
  })

  return { ensureBackend, handler, mintTicket, store }
}

function expectRequestHeaders(
  store: ReturnType<typeof createRemoteWsHeaderStore>,
  url: string,
  expected: Record<string, string> | undefined
) {
  const callback = vi.fn()

  applyRemoteRequestHeaders({ url, requestHeaders: { Origin: 'app://hermes' } }, callback, store.headersFor, new Map())

  expect(callback).toHaveBeenCalledOnce()
  expect(callback).toHaveBeenCalledWith(expected ? { requestHeaders: { Origin: 'app://hermes', ...expected } } : {})
}

function expectNoHeadersForNearbyUrls(store: ReturnType<typeof createRemoteWsHeaderStore>, exactUrl: string) {
  const exact = new URL(exactUrl)
  const unscoped = new URL(exact)
  unscoped.searchParams.delete('profile')
  const sibling = new URL(exact)
  sibling.pathname = '/api/ws/sibling'
  const otherProfile = new URL(exact)
  otherProfile.searchParams.set('profile', 'analysis')
  const otherCredential = new URL(exact)

  if (otherCredential.searchParams.has('ticket')) {
    otherCredential.searchParams.set('ticket', 'other-ticket')
  } else {
    otherCredential.searchParams.set('token', 'other-token')
  }

  const reordered = new URL(exact)
  const entries = [...reordered.searchParams.entries()].reverse()
  reordered.search = ''

  for (const [name, value] of entries) {
    reordered.searchParams.append(name, value)
  }

  for (const url of [unscoped, sibling, otherProfile, otherCredential, reordered]) {
    expect(store.headersFor(url.toString())).toEqual({})
    expectRequestHeaders(store, url.toString(), undefined)
  }
}

describe('registry gateway WebSocket headers', () => {
  it('evicts the least recently accessed exact URL', () => {
    const store = createRemoteWsHeaderStore(2)
    const firstUrl = 'wss://gateway.example/api/ws?token=first&profile=research'
    const secondUrl = 'wss://gateway.example/api/ws?token=second&profile=research'
    const thirdUrl = 'wss://gateway.example/api/ws?token=third&profile=research'

    store.remember(firstUrl, accessHeaders)
    store.remember(secondUrl, accessHeaders)
    expect(store.headersFor('wss://gateway.example/api/ws?token=missing&profile=research')).toEqual({})
    expect(store.headersFor(firstUrl)).toEqual(accessHeaders)

    store.remember(thirdUrl, accessHeaders)

    expect(store.headersFor(firstUrl)).toEqual(accessHeaders)
    expect(store.headersFor(secondUrl)).toEqual({})
    expect(store.headersFor(thirdUrl)).toEqual(accessHeaders)
  })

  it('updates headers without changing insertion recency', () => {
    const store = createRemoteWsHeaderStore(2)
    const firstUrl = 'wss://gateway.example/api/ws?token=first'
    const secondUrl = 'wss://gateway.example/api/ws?token=second'
    const thirdUrl = 'wss://gateway.example/api/ws?token=third'

    store.remember(firstUrl, { 'CF-Access-Client-Id': 'old-client-id' })
    store.remember(secondUrl, accessHeaders)
    store.remember(firstUrl, { 'CF-Access-Client-Id': 'updated-client-id' })
    store.remember(thirdUrl, accessHeaders)

    expect(store.headersFor(firstUrl)).toEqual({})
    expect(store.headersFor(secondUrl)).toEqual(accessHeaders)
    expect(store.headersFor(thirdUrl)).toEqual(accessHeaders)
  })

  it('token path binds headers to the exact profile scoped URL', async () => {
    const { ensureBackend, handler, mintTicket, store } = createHarness({
      authMode: 'token',
      baseUrl: 'https://gateway.example',
      wsUrl: 'wss://gateway.example/api/ws?token=secret&trace=one&profile=old',
      headers: accessHeaders,
      profile: 'research',
      sharedRemote: true
    })

    const result = await handler({ connectionId: 'remote-one', profile: 'research' })
    const expectedUrl = 'wss://gateway.example/api/ws?token=secret&trace=one&profile=research'

    expect(result).toBe(expectedUrl)
    expect(ensureBackend).toHaveBeenCalledWith('remote-one', 'research')
    expect(mintTicket).not.toHaveBeenCalled()
    expect(store.headersFor(result)).toEqual(accessHeaders)
    expectRequestHeaders(store, result, accessHeaders)
    expectNoHeadersForNearbyUrls(store, result)
  })

  it('OAuth path binds headers to the exact fresh profile scoped URL', async () => {
    const { handler, mintTicket, store } = createHarness({
      authMode: 'oauth',
      baseUrl: 'https://gateway.example',
      wsUrl: 'wss://gateway.example/api/ws?ticket=stale',
      headers: accessHeaders,
      profile: 'research',
      sharedRemote: true
    })

    const result = await handler({ connectionId: 'cloud-one', profile: 'research' })
    const expectedUrl = 'wss://gateway.example/api/ws?region=us&ticket=fresh-ticket&profile=research'

    expect(result).toBe(expectedUrl)
    expect(mintTicket).toHaveBeenCalledOnce()
    expect(mintTicket).toHaveBeenCalledWith('https://gateway.example', accessHeaders)
    expect(store.headersFor(result)).toEqual(accessHeaders)
    expectRequestHeaders(store, result, accessHeaders)
    expectNoHeadersForNearbyUrls(store, result)
  })

  it('sharedRemote false preserves the original URL and exact header behavior', async () => {
    const { handler, store } = createHarness({
      authMode: 'token',
      baseUrl: 'https://gateway.example',
      wsUrl: 'wss://gateway.example/api/ws?trace=one&token=secret',
      headers: accessHeaders,
      profile: 'research',
      sharedRemote: false
    })

    const result = await handler({ connectionId: 'remote-one', profile: 'research' })

    expect(result).toBe('wss://gateway.example/api/ws?trace=one&token=secret')
    expect(store.headersFor(result)).toEqual(accessHeaders)
    expectRequestHeaders(store, result, accessHeaders)
    expect(store.headersFor('wss://gateway.example/api/ws?token=secret&trace=one')).toEqual({})
  })
})

describe('OAuth login and registry extra headers', () => {
  it('applies Connections extra headers to /login, not only an exact WebSocket URL', () => {
    const sources = collectRemoteHeaderSources({
      connections: [{ kind: 'local' }, { kind: 'remote', url: 'https://gateway.example', headers: accessHeaders }],
      v1Remote: { url: 'https://other.example', headers: { 'CF-Access-Client-Id': 'v1-only' } }
    })

    expect(resolveRemoteRequestHeaders('https://gateway.example/login', { sources })).toEqual(accessHeaders)
    expect(resolveRemoteRequestHeaders('https://gateway.example/api/status', { sources })).toEqual(accessHeaders)
    expect(oauthLoginLoadUrlOptions(accessHeaders)).toEqual({
      extraHeaders: 'CF-Access-Client-Id: client-id\nCF-Access-Client-Secret: client-secret'
    })
    expect(resolveRemoteRequestHeaders('https://other.example/login', { sources })).toEqual({
      'CF-Access-Client-Id': 'v1-only'
    })
  })

  it('keeps configured headers inside their gateway path and strips them from redirects outside it', () => {
    const listeners = []
    let completed = details => details

    const oauthSession = {
      webRequest: {
        onBeforeSendHeaders: listener => {
          listeners.push(listener)
        },
        onCompleted: listener => {
          completed = listener
        }
      }
    }

    const scopedHeaders = { ...accessHeaders, 'X-Api-Key': 'configured-secret' }

    const sources = collectRemoteHeaderSources({
      connections: [{ kind: 'remote', url: 'https://gateway.example/hermes', headers: scopedHeaders }]
    })

    attachRemoteRequestHeaderListener(oauthSession, url => resolveRemoteRequestHeaders(url, { sources }))

    const initial = vi.fn()
    listeners[0](
      { id: 1, url: 'https://gateway.example/hermes/login', requestHeaders: { Origin: 'app://hermes' } },
      initial
    )
    expect(initial).toHaveBeenCalledWith({
      requestHeaders: { Origin: 'app://hermes', ...scopedHeaders }
    })

    const sameScope = vi.fn()
    listeners[0](
      {
        id: 1,
        url: 'https://gateway.example/hermes/ready',
        requestHeaders: { Origin: 'app://hermes', Cookie: 'session=live', ...scopedHeaders }
      },
      sameScope
    )
    expect(sameScope).toHaveBeenCalledWith({
      requestHeaders: { Origin: 'app://hermes', Cookie: 'session=live', ...scopedHeaders }
    })

    for (const [id, redirectUrl] of [
      [2, 'https://gateway.example/login'],
      [3, 'https://identity.example/callback']
    ] as const) {
      listeners[0]({ id, url: 'https://gateway.example/hermes/start', requestHeaders: {} }, vi.fn())

      const redirected = vi.fn()
      listeners[0](
        {
          id,
          url: redirectUrl,
          requestHeaders: {
            Origin: 'app://hermes',
            Cookie: 'idp-session=live',
            'x-api-key': scopedHeaders['X-Api-Key'],
            'cf-access-client-id': scopedHeaders['CF-Access-Client-Id'],
            'CF-Access-Client-Secret': scopedHeaders['CF-Access-Client-Secret']
          }
        },
        redirected
      )
      expect(redirected).toHaveBeenCalledWith({
        requestHeaders: { Origin: 'app://hermes', Cookie: 'idp-session=live' }
      })
    }

    // Stripping is keyed on the request we injected into, not on header
    // values: an unrelated request carrying the same name — even the same
    // value — passes through untouched.
    completed({ id: 1 })

    for (const [id, value] of [
      [4, 'identity-provider-key'],
      [4, scopedHeaders['X-Api-Key']],
      [1, scopedHeaders['X-Api-Key']]
    ] as const) {
      const unrelated = vi.fn()
      listeners[0]({ id, url: 'https://identity.example/token', requestHeaders: { 'X-Api-Key': value } }, unrelated)
      expect(unrelated).toHaveBeenCalledWith({})
    }
  })
})
