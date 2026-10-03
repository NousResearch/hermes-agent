import { describe, expect, it, vi } from 'vitest'

import {
  attachRemoteRequestHeaderListener,
  createRegistryGatewayWsUrlHandler,
  createRemoteWsHeaderStore,
  resolveRemoteRequestHeaders
} from './remote-ws-headers'

function harness(rendererOrigin = 'http://127.0.0.1:47891') {
  const store = createRemoteWsHeaderStore()
  let listener: any
  attachRemoteRequestHeaderListener(
    {
      webRequest: {
        onBeforeSendHeaders: fn => {
          listener = fn
        }
      }
    },
    store.headersFor,
    rendererOrigin
  )

  const send = (url: string, requestHeaders: Record<string, string>) => {
    const callback = vi.fn()
    listener({ url, requestHeaders }, callback)

    return callback.mock.calls[0][0]
  }

  return { store, send }
}

const base = 'https://gateway.example:9443'
const wsUrl = 'wss://gateway.example:9443/api/ws?ticket=fresh&profile=research'

describe('native gateway Origin isolation', () => {
  it('composes exact Origin stamping with case-insensitive proxy replacement and request-scoped redirect cleanup', () => {
    const store = createRemoteWsHeaderStore()
    const proxy = { 'X-Api-Key': 'synthetic-proxy-key', 'CF-Access-Client-Id': 'synthetic-client' }
    const sources = [{ url: base, headers: proxy }]
    store.remember(wsUrl)
    const before = vi.fn()
    const completed = vi.fn()
    const errored = vi.fn()
    attachRemoteRequestHeaderListener(
      { webRequest: { onBeforeSendHeaders: before, onCompleted: completed, onErrorOccurred: errored } },
      url => resolveRemoteRequestHeaders(url, { exactHeaders: store.headersFor(url), sources }),
      'http://127.0.0.1:47891'
    )
    const send = (id: number, url: string, requestHeaders: Record<string, string>) => {
      const callback = vi.fn()
      before.mock.calls[0][0]({ id, url, requestHeaders }, callback)
      expect(callback).toHaveBeenCalledOnce()
      return callback.mock.calls[0][0]
    }

    for (const [id, origin, native] of [
      [1, 'http://127.0.0.1:47891', true],
      [2, 'http://127.0.0.1:47892', false],
      [3, 'https://foreign.example', false]
    ] as const) {
      const initial = send(id, wsUrl, { origin, 'x-api-key': 'stale', 'cf-access-client-id': 'stale', Cookie: 'keep' })
      expect(initial).toEqual({ requestHeaders: { Origin: native ? 'null' : origin, ...proxy, Cookie: 'keep' } })
      // The next hop is inside proxy scope, but not the exact issued WS URL.
      const nearby = send(id, wsUrl.replace('fresh', 'different'), initial.requestHeaders)
      expect(nearby).toEqual({ requestHeaders: { ...proxy, Cookie: 'keep', ...(native ? {} : { Origin: origin }) } })
      const redirect = send(id, 'https://identity.example/login', nearby.requestHeaders)
      expect(redirect).toEqual({ requestHeaders: { Cookie: 'keep', ...(native ? {} : { Origin: origin }) } })
      // An unrelated request with the same values must never be stripped.
      expect(send(id + 100, 'https://identity.example/login', initial.requestHeaders)).toEqual({})
    }
    // A direct out-of-scope redirect strips the stamp as well as credentials.
    for (const [id, forget] of [
      [4, completed],
      [5, errored]
    ] as const) {
      const initial = send(id, wsUrl, { Origin: 'http://127.0.0.1:47891', Cookie: 'keep' })
      const carried = Object.fromEntries(
        Object.entries<string>(initial.requestHeaders).map(([name, value]) => [name.toLowerCase(), value])
      )
      expect(send(id, 'https://identity.example/login', carried)).toEqual({ requestHeaders: { cookie: 'keep' } })
      send(id, wsUrl, { Origin: 'http://127.0.0.1:47891' })
      forget.mock.calls[0][0]({ id })
      expect(send(id, 'https://identity.example/login', carried)).toEqual({})
    }
  })
  it('stamps only exact issued native WebSocket requests while preserving proxy credentials and foreign Origins', () => {
    const { store, send } = harness()
    const accessHeaders = { 'CF-Access-Client-Id': 'id', 'CF-Access-Client-Secret': 'secret' }
    store.remember(wsUrl, accessHeaders)

    for (const origin of ['http://127.0.0.1:47891', 'null', 'file://', 'app://hermes', '']) {
      expect(send(wsUrl, { origin, Cookie: 'existing' })).toEqual({
        requestHeaders: { Origin: 'null', Cookie: 'existing', ...accessHeaders }
      })
    }

    for (const origin of [
      'https://untrusted.example',
      'http://127.0.0.1.evil.example:47891',
      'not-an-origin',
      'http://127.0.0.1:47892',
      'http://localhost:47891',
      'http://localhost:5174',
      'https://127.0.0.1:47891',
      'http://127.0.0.1:47891/path',
      'http://user@127.0.0.1:47891',
      'file:///untrusted.html',
      'app://untrusted'
    ]) {
      expect(send(wsUrl, { origin })).toEqual({ requestHeaders: { Origin: origin, ...accessHeaders } })
    }

    for (const url of [
      base + '/api/auth/ws-ticket',
      wsUrl.replace('fresh', 'other'),
      wsUrl.replace('research', 'other'),
      wsUrl.replace('gateway.', 'other.'),
      wsUrl.replace('/api/ws', '/api/events')
    ]) {
      expect(send(url, { Origin: 'http://127.0.0.1:47891' })).toEqual({})
    }

    store.remember(base + '/api/ws?ticket=fresh')
    expect(send(base + '/api/ws?ticket=fresh', { Origin: 'http://127.0.0.1:47891' })).toEqual({})

    // The active dev URL may use another hostname/port. File mode must not
    // accidentally retain dev's web-origin privilege.
    for (const rendererOrigin of ['http://localhost:5174', 'https://[::1]:5174', 'null']) {
      const active = harness(rendererOrigin)
      for (const url of [wsUrl, 'ws://127.0.0.1:9119/api/ws?token=local', 'ws://127.0.0.1:49200/api/ws?token=ssh']) {
        active.store.remember(url)
        expect(active.send(url, { Origin: rendererOrigin })).toEqual({ requestHeaders: { Origin: 'null' } })
        expect(active.send(url, { Origin: 'http://127.0.0.1:47891' })).toEqual({
          requestHeaders: { Origin: 'http://127.0.0.1:47891' }
        })
      }
    }
  })

  it('registers each fresh OAuth reconnect without extra headers on the final profile-scoped URL', async () => {
    const { store, send } = harness()
    let n = 0
    const mintTicket = vi.fn(async () => `fresh-${++n}`)

    const handler = createRegistryGatewayWsUrlHandler({
      ensureBackend: async () => ({ authMode: 'oauth', baseUrl: base, wsUrl, profile: 'research', sharedRemote: true }),
      mintTicket,
      buildTicketUrl: (_, ticket) => `wss://gateway.example:9443/api/ws?ticket=${ticket}`,
      rememberHeaders: store.remember
    })

    const first = await handler({ connectionId: 'remote-one', profile: 'research' })
    const second = await handler({ connectionId: 'remote-one', profile: 'research' })
    expect(first).not.toBe(second)
    expect(mintTicket).toHaveBeenCalledTimes(2)

    for (const url of [first, second]) {
      expect(new URL(url).searchParams.get('profile')).toBe('research')
      expect(send(url, { Origin: 'http://127.0.0.1:47891' })).toEqual({ requestHeaders: { Origin: 'null' } })
      expect(send(url.replace('research', 'other'), { Origin: 'http://127.0.0.1:47891' })).toEqual({})
    }
  })
})
