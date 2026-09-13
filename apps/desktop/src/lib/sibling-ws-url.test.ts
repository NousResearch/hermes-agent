import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { resolveSiblingWsUrl } from './sibling-ws-url'

// A sibling stream (voice PCM, Bot Screen RFB) must dial the SAME (connection,
// profile) backend chat uses. The bare v1 getConnection pair answers for the
// local primary — the wrong machine when a registry remote rides over a local
// install — so registry routes must go through the *For bridges.
describe('resolveSiblingWsUrl', () => {
  const remoteWsUrl = 'wss://gateway.example/api/ws?ticket=fresh'
  const localWsUrl = 'ws://127.0.0.1:5151/api/ws?token=local'

  let getConnection: ReturnType<typeof vi.fn>
  let getConnectionFor: ReturnType<typeof vi.fn>
  let getGatewayWsUrl: ReturnType<typeof vi.fn>
  let getGatewayWsUrlFor: ReturnType<typeof vi.fn>

  beforeEach(() => {
    getConnection = vi.fn(async () => ({ authMode: 'token', baseUrl: 'http://127.0.0.1:5151', wsUrl: localWsUrl }))
    getConnectionFor = vi.fn(async () => ({ authMode: 'token', baseUrl: 'https://gateway.example', wsUrl: remoteWsUrl }))
    getGatewayWsUrl = vi.fn(async () => ({ ok: true, wsUrl: localWsUrl }))
    getGatewayWsUrlFor = vi.fn(async () => ({ ok: true, wsUrl: remoteWsUrl }))
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { getConnection, getConnectionFor, getGatewayWsUrl, getGatewayWsUrlFor }
    })
  })

  afterEach(() => {
    Reflect.deleteProperty(window, 'hermesDesktop')
  })

  it('routes a registry-scoped profile through the *For bridges and swaps only the path', async () => {
    const url = await resolveSiblingWsUrl({ connectionId: 'gw-tailscale', profile: 'research' }, '/api/display/ws')

    expect(url).toBe('wss://gateway.example/api/display/ws?ticket=fresh')
    expect(getConnectionFor).toHaveBeenCalledWith({ connectionId: 'gw-tailscale', profile: 'research' })
    expect(getConnection).not.toHaveBeenCalled()
    expect(getGatewayWsUrl).not.toHaveBeenCalled()
  })

  it('strips the spent gateway credential when the sibling route authenticates itself', async () => {
    const url = new URL(await resolveSiblingWsUrl({ profile: null }, 'api/display/ws', { stripGatewayCredential: true }))

    expect(url.origin + url.pathname).toBe('ws://127.0.0.1:5151/api/display/ws')
    expect(url.searchParams.has('token')).toBe(false)
    expect(url.searchParams.has('ticket')).toBe(false)
  })
})
