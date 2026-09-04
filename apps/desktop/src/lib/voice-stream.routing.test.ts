import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { setApiRequestConnection, setApiRequestProfile } from '@/hermes'

import { resolveTranscribeStreamUrl } from './voice-stream'

// The transcribe-stream WebSocket must dial the ACTIVE (connection, profile)
// backend — the same routing contract as resolveSpeakStreamUrl: when a
// registry remote rides over a local install, live dictation audio must reach
// the REMOTE gateway (where the profile's STT provider runs), not the local
// machine's web server.
describe('resolveTranscribeStreamUrl', () => {
  const remoteWsUrl = 'wss://gateway.example/api/ws?ticket=fresh'
  const localWsUrl = 'ws://127.0.0.1:5151/api/ws?token=local'

  let getConnection: ReturnType<typeof vi.fn>
  let getConnectionFor: ReturnType<typeof vi.fn>
  let getGatewayWsUrl: ReturnType<typeof vi.fn>
  let getGatewayWsUrlFor: ReturnType<typeof vi.fn>

  beforeEach(() => {
    getConnection = vi.fn(async () => ({ authMode: 'token', baseUrl: 'http://127.0.0.1:5151', wsUrl: localWsUrl }))

    getConnectionFor = vi.fn(async () => ({
      authMode: 'token',
      baseUrl: 'https://gateway.example',
      wsUrl: remoteWsUrl
    }))

    getGatewayWsUrl = vi.fn(async () => ({ ok: true, wsUrl: localWsUrl }))
    getGatewayWsUrlFor = vi.fn(async () => ({ ok: true, wsUrl: remoteWsUrl }))

    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { getConnection, getConnectionFor, getGatewayWsUrl, getGatewayWsUrlFor }
    })
  })

  afterEach(() => {
    setApiRequestConnection(null)
    setApiRequestProfile(null)
    Reflect.deleteProperty(window, 'hermesDesktop')
  })

  it('resolves through the registry (connection, profile) bridges when a registry connection is active', async () => {
    setApiRequestConnection('gw-tailscale')
    setApiRequestProfile('research')

    const url = await resolveTranscribeStreamUrl()

    expect(url).toContain('wss://gateway.example')
    expect(url).toContain('/api/audio/transcribe-stream')
    expect(getConnectionFor).toHaveBeenCalledWith({ connectionId: 'gw-tailscale', profile: 'research' })
    expect(getGatewayWsUrlFor).toHaveBeenCalledWith({ connectionId: 'gw-tailscale', profile: 'research' })
    // The v1 primary path must NOT be consulted — that's the local machine.
    expect(getConnection).not.toHaveBeenCalled()
    expect(getGatewayWsUrl).not.toHaveBeenCalled()
  })

  it('keeps a backend-namespaced profile query param when one is already present', async () => {
    setApiRequestConnection('gw-tailscale')
    setApiRequestProfile('research')
    getGatewayWsUrlFor.mockResolvedValue({
      ok: true,
      wsUrl: 'wss://gateway.example/api/ws?profile=backend-ns&ticket=t'
    })

    const url = await resolveTranscribeStreamUrl()

    expect(url).toContain('profile=backend-ns')
    expect(url).not.toContain('profile=research')
  })

  it('falls back to the v1 primary backend when no registry connection is active', async () => {
    const url = await resolveTranscribeStreamUrl()

    expect(url).toContain('127.0.0.1:5151')
    expect(url).toContain('/api/audio/transcribe-stream')
    expect(getConnection).toHaveBeenCalledWith(null)
    expect(getGatewayWsUrlFor).not.toHaveBeenCalled()
  })

  it('returns null when the gateway URL is not the /api/ws shape', async () => {
    getGatewayWsUrl.mockResolvedValue({ ok: true, wsUrl: 'ws://127.0.0.1:5151/other' })

    expect(await resolveTranscribeStreamUrl()).toBeNull()
  })

  it('returns null without desktop bridges', async () => {
    Reflect.deleteProperty(window, 'hermesDesktop')

    expect(await resolveTranscribeStreamUrl()).toBeNull()
  })
})
