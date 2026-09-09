import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { setApiRequestConnection, setApiRequestProfile } from '@/hermes'

import { resolveSpeakStreamUrl } from './voice-playback'

// The speak-stream WebSocket must dial the ACTIVE (connection, profile)
// backend — the same one chat and every REST audio call use. Before this
// contract was pinned it resolved through the bare v1 getConnection path,
// so a registry remote riding over a local install synthesized replies with
// the LOCAL machine's (often unconfigured) TTS while chat correctly went
// remote (desktop-remote voice report, Aug 2026).
describe('resolveSpeakStreamUrl', () => {
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
    vi.useRealTimers()
  })

  it('resolves through the registry (connection, profile) bridges when a registry connection is active', async () => {
    setApiRequestConnection('gw-tailscale')
    setApiRequestProfile('research')

    const url = await resolveSpeakStreamUrl()

    expect(url).toContain('wss://gateway.example')
    expect(url).toContain('/api/audio/speak-stream')
    expect(getConnectionFor).toHaveBeenCalledWith({ connectionId: 'gw-tailscale', profile: 'research' })
    expect(getGatewayWsUrlFor).toHaveBeenCalledWith({ connectionId: 'gw-tailscale', profile: 'research' })
    // The v1 primary path must NOT be consulted — that's the local machine.
    expect(getConnection).not.toHaveBeenCalled()
    expect(getGatewayWsUrl).not.toHaveBeenCalled()
  })

  it('keeps the legacy profile path byte-identical when no registry connection is active', async () => {
    setApiRequestProfile('coder')

    const url = await resolveSpeakStreamUrl()

    expect(url).toContain('ws://127.0.0.1:5151')
    expect(url).toContain('/api/audio/speak-stream')
    expect(url).toContain('profile=coder')
    expect(getConnection).toHaveBeenCalledWith('coder')
    expect(getConnectionFor).not.toHaveBeenCalled()
  })

  it('preserves a backend-namespace profile already minted into the ws URL', async () => {
    // SSH remoteProfile aliasing / sharedRemote scoping: the registry mint
    // writes the BACKEND's profile name into the URL. The desktop-side
    // routing alias must not overwrite it.
    setApiRequestConnection('gw-ssh')
    setApiRequestProfile('mara')
    getGatewayWsUrlFor.mockResolvedValue({
      ok: true,
      wsUrl: 'wss://gateway.example/api/ws?ticket=fresh&profile=default'
    })

    const url = await resolveSpeakStreamUrl()

    expect(url).toContain('profile=default')
    expect(url).not.toContain('profile=mara')
  })

  it('falls back to the plain connection descriptor when the *For bridges are absent (older main)', async () => {
    setApiRequestConnection('gw-tailscale')
    setApiRequestProfile('research')
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { getConnection, getGatewayWsUrl }
    })

    const url = await resolveSpeakStreamUrl()

    // Best available answer without the bridges: the profile-scoped pool
    // descriptor's own wsUrl (no cross-scope re-mint).
    expect(url).toContain('/api/audio/speak-stream')
    expect(getConnection).toHaveBeenCalledWith('research')
  })

  it('resolves to null instead of hanging forever when getConnection() wedges (#93454)', async () => {
    // desktop.getConnection/getConnectionFor/resolveGatewayWsUrl are IPC
    // round-trips into the main process with no timeout of their own. A
    // wedged main-process round-trip otherwise hangs voice mode's "speaking"
    // state forever instead of falling back to playSpeechText.
    vi.useFakeTimers()
    setApiRequestProfile('coder')
    getConnection.mockImplementation(() => new Promise(() => undefined))

    const pending = resolveSpeakStreamUrl()

    await vi.advanceTimersByTimeAsync(20_000)

    await expect(pending).resolves.toBeNull()
  })
})
