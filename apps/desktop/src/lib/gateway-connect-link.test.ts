import { describe, expect, it } from 'vitest'

import { resolveDeepLinkAction } from './deeplink-routes'
import { parseGatewayConnectRequest } from './gateway-connect-link'

describe('gateway connection handoff', () => {
  it('normalizes a gateway address without changing its cluster path', () => {
    expect(
      resolveDeepLinkAction({
        kind: 'gateway',
        name: 'connect',
        params: { url: 'https://ACTUAL.inc/api/hermes/clusters/cluster-1/', name: 'Studio' }
      })
    ).toEqual({
      type: 'gateway-connect',
      request: { url: 'https://actual.inc/api/hermes/clusters/cluster-1', name: 'Studio' }
    })
  })

  it.each([
    'file:///etc/passwd',
    'javascript:alert(1)',
    'http://example.com',
    'https://user:secret@example.com',
    'https://example.com?token=secret',
    'https://example.com#secret'
  ])('rejects unsafe addresses: %s', url => {
    expect(parseGatewayConnectRequest({ url })).toBeNull()
  })

  it('does not accept credentials or commands in the handoff', () => {
    expect(parseGatewayConnectRequest({ url: 'https://example.com', token: 'secret' })).toBeNull()
    expect(parseGatewayConnectRequest({ url: 'https://example.com', name: 'bad\nname' })).toBeNull()
  })
})
