import { describe, expect, it, vi } from 'vitest'

import {
  applyRemoteRequestHeaders,
  createRegistryGatewayWsUrlHandler,
  createRemoteWsHeaderStore,
  type RegistryGatewayWsConnection
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

  applyRemoteRequestHeaders({ url, requestHeaders: { Origin: 'app://hermes' } }, callback, store.headersFor)

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

  // The registry path is the reconnect path, so it is where a forwarded proxy
  // session has to be bound (gateway-ws-cookie.ts). Pin that rememberHeaders
  // receives the resolved connection alongside the FINAL url, and is awaited.
  it('hands the resolved connection and final URL to rememberHeaders', async () => {
    const connection: RegistryGatewayWsConnection = {
      authMode: 'oauth',
      baseUrl: 'https://gateway.example',
      wsUrl: 'wss://gateway.example/api/ws?ticket=stale',
      headers: accessHeaders,
      profile: 'research',
      sharedRemote: true
    }

    const seen: Array<{ connection?: RegistryGatewayWsConnection; wsUrl: string }> = []
    let resolveRemember: () => void = () => undefined
    let signalEntered: () => void = () => undefined
    const remembered = new Promise<void>(resolve => {
      resolveRemember = resolve
    })
    const entered = new Promise<void>(resolve => {
      signalEntered = resolve
    })

    const handler = createRegistryGatewayWsUrlHandler({
      ensureBackend: vi.fn(async () => connection),
      mintTicket: vi.fn(async () => 'fresh-ticket'),
      buildTicketUrl: (baseUrl, ticket) => `${baseUrl.replace(/^https:/, 'wss:')}/api/ws?ticket=${ticket}`,
      rememberHeaders: async (wsUrl, _headers, resolved) => {
        seen.push({ connection: resolved, wsUrl })
        signalEntered()
        await remembered
      }
    })

    const pending = handler({ connectionId: 'cloud-one', profile: 'research' })
    let settled = false
    void pending.then(() => {
      settled = true
    })

    await entered

    expect(seen).toHaveLength(1)

    // The url must not reach the renderer before its cookie authority is
    // registered, so the handler stays pending while rememberHeaders does.
    await Promise.resolve()
    expect(settled).toBe(false)

    resolveRemember()

    const result = await pending
    const expectedUrl = 'wss://gateway.example/api/ws?ticket=fresh-ticket&profile=research'

    expect(result).toBe(expectedUrl)
    expect(seen[0].wsUrl).toBe(expectedUrl)
    expect(seen[0].connection?.baseUrl).toBe('https://gateway.example')
    expect(seen[0].connection?.authMode).toBe('oauth')
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
