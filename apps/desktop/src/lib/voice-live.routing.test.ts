import { afterEach, describe, expect, it, vi } from 'vitest'

import { setApiRequestConnection, setApiRequestProfile } from '@/hermes'

import { fetchVoiceLiveStatus, type VoiceLiveHandlers, VoiceLiveSession } from './voice-live'

// A GPT-Live session is created on the chat OWNER's (connection, profile) —
// the Bot that owns the chat — never on the window's active scope, mirroring
// the TTS legs (#117014): two Bots with different voices configured must get
// their own voices. The status probe follows the same rule so a Bot tile
// mounts the engine the Bot's own profile selected.

const handlers: VoiceLiveHandlers = {
  onClosed: () => undefined,
  onDelegation: () => undefined,
  onError: () => undefined
}

function installApi() {
  const api = vi.fn(async (_request: unknown) => ({ ok: true, session: { id: 's1' }, transport: { sdp: 'answer', type: 'answer' } }))

  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { api } })

  return api
}

/** Enough of RTCPeerConnection/getUserMedia for `start()` to reach the POST. */
function installWebRTC() {
  const channel = { addEventListener: () => undefined, readyState: 'open' as const, send: () => undefined }

  const PeerConnection = class {
    addEventListener = () => undefined
    connectionState = 'new'
    createDataChannel = () => channel
    createOffer = async () => ({})
    iceGatheringState = 'complete'
    localDescription = { sdp: 'v=0\r\n' }
    setLocalDescription = async () => undefined
    setRemoteDescription = async () => undefined
  }

  Object.defineProperty(globalThis, 'RTCPeerConnection', { configurable: true, value: PeerConnection })
  Object.defineProperty(window.navigator, 'mediaDevices', {
    configurable: true,
    value: { getUserMedia: async () => ({ getAudioTracks: () => [] }) }
  })
}

afterEach(() => {
  setApiRequestConnection(null)
  setApiRequestProfile(null)
  Reflect.deleteProperty(window, 'hermesDesktop')
  Reflect.deleteProperty(globalThis, 'RTCPeerConnection')
  Reflect.deleteProperty(window.navigator, 'mediaDevices')
})

describe('GPT-Live owner routing', () => {
  it('creates the session on the chat owner (connection, profile) ahead of the active scope', async () => {
    installWebRTC()
    const api = installApi()
    setApiRequestConnection('gw-active')
    setApiRequestProfile('research')

    const session = new VoiceLiveSession(handlers, { connectionId: 'gw-bots', profile: 'bot-adam' })
    await session.start([])

    expect(api).toHaveBeenCalledTimes(1)
    const request = api.mock.calls[0][0] as { connectionId?: string; path: string; priority?: string; profile?: string }

    expect(request.path).toBe('/api/audio/voice-live/session')
    expect(request.profile).toBe('bot-adam')
    expect(request.connectionId).toBe('gw-bots')
    expect(request.priority).toBe('foreground')
  })

  it('keeps the active scope verbatim for an ownerless chat', async () => {
    installWebRTC()
    const api = installApi()
    setApiRequestConnection('gw-active')
    setApiRequestProfile('research')

    const session = new VoiceLiveSession(handlers, null)
    await session.start([])

    const request = api.mock.calls[0][0] as { connectionId?: string; path: string; priority?: string; profile?: string }

    expect(request.path).toBe('/api/audio/voice-live/session')
    expect(request.profile).toBe('research')
    expect(request.connectionId).toBe('gw-active')
    expect(request.priority).toBeUndefined()
  })

  it('probes the mode status on the owner scope too', async () => {
    const api = installApi()
    setApiRequestConnection('gw-active')
    setApiRequestProfile('research')

    await fetchVoiceLiveStatus({ connectionId: 'gw-bots', profile: 'bot-adam' })

    const request = api.mock.calls[0][0] as { connectionId?: string; path: string; priority?: string; profile?: string }

    expect(request.path).toBe('/api/audio/voice-live/status')
    expect(request.profile).toBe('bot-adam')
    expect(request.connectionId).toBe('gw-bots')
    expect(request.priority).toBe('foreground')
  })
})
