import { describe, expect, it, vi } from 'vitest'

import {
  attachRemoteRequestHeaderListener,
  createRegistryGatewayWsUrlHandler,
  createRemoteWsHeaderStore
} from './remote-ws-headers'

function harness() {
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
    store.headersFor
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
  it('stamps only exact issued native WebSocket requests while preserving proxy credentials and foreign Origins', () => {
    const { store, send } = harness()
    const accessHeaders = { 'CF-Access-Client-Id': 'id', 'CF-Access-Client-Secret': 'secret' }
    store.remember(wsUrl, accessHeaders)

    for (const origin of ['http://127.0.0.1:47891', 'http://localhost:5174', 'null']) {
      expect(send(wsUrl, { origin, Cookie: 'existing' })).toEqual({
        requestHeaders: { Origin: 'null', Cookie: 'existing', ...accessHeaders }
      })
    }

    for (const origin of ['https://untrusted.example', 'http://127.0.0.1.evil.example:47891', 'not-an-origin']) {
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
