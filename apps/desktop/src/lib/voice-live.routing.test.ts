import { afterEach, describe, expect, it, vi } from 'vitest'

import { setApiRequestConnection, setApiRequestProfile } from '@/hermes'

import { type VoiceLiveHandlers, VoiceLiveSession } from './voice-live'

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
  const api = vi.fn(async (_request: unknown) => ({
    capture_status: undefined as 'recording' | 'unavailable' | undefined,
    ok: true,
    session: { id: 's1' },
    transport: { sdp: 'answer', type: 'answer' }
  }))

  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { api } })

  return api
}

/** Enough of RTCPeerConnection/getUserMedia for `start()` to reach the POST. */
function installWebRTC() {
  const channel = new EventTarget()
  Object.assign(channel, { readyState: 'open', send: () => undefined })

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

  return channel
}

afterEach(() => {
  setApiRequestConnection(null)
  setApiRequestProfile(null)
  Reflect.deleteProperty(window, 'hermesDesktop')
  Reflect.deleteProperty(globalThis, 'RTCPeerConnection')
  Reflect.deleteProperty(window.navigator, 'mediaDevices')
})

describe('GPT-Live owner routing', () => {
  it('distinguishes a real zero timestamp from an omitted start time', async () => {
    const channel = installWebRTC()
    installApi()
    const onTranscript = vi.fn()
    const session = new VoiceLiveSession({ ...handlers, onTranscript })
    await session.start([])

    for (const timing of [{ end_ms: 400 }, { end_ms: 400, start_ms: 0 }]) {
      channel.dispatchEvent(
        new MessageEvent('message', {
          data: JSON.stringify({ ...timing, delta: 'Hello.', type: 'session.input_transcript.delta' })
        })
      )
    }

    expect(onTranscript.mock.calls.map(([fragment]) => fragment.timingMissing)).toEqual([true, undefined])
  })

  it('creates the session on the chat owner (connection, profile) ahead of the active scope', async () => {
    installWebRTC()
    const api = installApi()
    setApiRequestConnection('gw-active')
    setApiRequestProfile('research')

    const session = new VoiceLiveSession(handlers, { connectionId: 'gw-bots', profile: 'bot-adam' })
    // Legacy gateways don't send capture_status; they still negotiate audio.
    await expect(session.start([], 'stored-chat')).resolves.toBe('unavailable')

    expect(api).toHaveBeenCalledTimes(1)
    const request = api.mock.calls[0][0] as { connectionId?: string; path: string; priority?: string; profile?: string }

    expect(request.path).toBe('/api/audio/voice-live/session')
    expect(request.profile).toBe('bot-adam')
    expect(request.connectionId).toBe('gw-bots')
    expect(request.priority).toBe('foreground')
    expect(request).toHaveProperty('body.session_id', 'stored-chat')

    // Ownerless chats pin the active owner at call start.
    const ownerless = new VoiceLiveSession(handlers, null)
    await ownerless.start([])
    const plain = api.mock.calls[1][0] as { connectionId?: string; path: string; priority?: string; profile?: string }
    expect(plain.path).toBe('/api/audio/voice-live/session')
    expect(plain.profile).toBe('research')
    expect(plain.connectionId).toBe('gw-active')
    expect(plain.priority).toBe('foreground')
    expect(plain).not.toHaveProperty('body.session_id')
  })

  it('keeps the negotiated call on an explicitly connection-less owner', async () => {
    installWebRTC()
    const api = installApi()
    setApiRequestConnection('other-gateway')
    setApiRequestProfile('other-profile')

    await new VoiceLiveSession(handlers).start([], 'fresh-chat', { connectionId: null, profile: 'legacy-worker' })

    expect(api).toHaveBeenCalledExactlyOnceWith(expect.objectContaining({
      body: expect.objectContaining({ session_id: 'fresh-chat' }),
      profile: 'legacy-worker'
    }))
    expect(api.mock.calls[0][0]).not.toHaveProperty('connectionId')
  })

  it('drops replayed transcript IDs before captions and delegation, but keeps repeated words', async () => {
    const api = installApi()
    const onTranscript = vi.fn()
    const onDelegation = vi.fn()
    const user = { delta: 'Go ', end_ms: 100, event_id: 'USER-1', start_ms: 0, type: 'session.input_transcript.delta' }
    const assistant = { delta: 'OK.', event_id: 'ASSISTANT-1', type: 'session.output_transcript.delta' }
    const noId = { delta: 'go ', type: 'session.input_transcript.delta' }

    for (const id of ['call-one', 'call-two']) {
      const channel = installWebRTC()
      api.mockResolvedValueOnce({ capture_status: 'recording', ok: true, session: { id }, transport: { sdp: 'answer', type: 'answer' } })
      await new VoiceLiveSession({ ...handlers, onDelegation, onTranscript }).start([])

      for (const event of [assistant, assistant, user, user, { ...user, event_id: 'USER-2' }, noId, noId,
        { delegation: { id: 'task' }, type: 'session.delegation.created' }]) {
        channel.dispatchEvent(new MessageEvent('message', { data: JSON.stringify(event) }))
      }
    }

    expect(onDelegation).toHaveBeenCalledTimes(2)

    for (const [, context] of onDelegation.mock.calls) {
      expect(context.map((part: { event_id?: string; text: string }) => [part.event_id, part.text])).toEqual([
        ['ASSISTANT-1', 'OK.'], ['USER-1', 'Go '], ['USER-2', 'Go '], [undefined, 'go '], [undefined, 'go ']
      ])
    }

    expect(onTranscript).toHaveBeenCalledTimes(10)
  })

  it('hands each vendor delegation off once with an identity scoped to the call', async () => {
    const api = installApi()
    const onDelegation = vi.fn()

    for (const callId of ['call-one', 'call-two', 'call-one']) {
      const channel = installWebRTC()
      api.mockResolvedValueOnce({
        capture_status: 'recording',
        ok: true,
        session: { id: callId },
        transport: { sdp: 'answer', type: 'answer' }
      })
      const session = new VoiceLiveSession({ ...handlers, onDelegation })

      await expect(session.start([], 'stored-chat')).resolves.toBe('recording')

      for (const id of ['one', 'two', 'one']) {
        channel.dispatchEvent(new MessageEvent('message', {
          data: JSON.stringify({ delegation: { id }, type: 'session.delegation.created' })
        }))
      }
    }

    expect(onDelegation.mock.calls.map(([id]) => id)).toEqual(['one', 'two', 'one', 'two', 'one', 'two'])
    const ids = onDelegation.mock.calls.map(([, , submissionId]) => submissionId)

    expect(onDelegation.mock.calls.map(([, , , callId]) => callId)).toEqual([
      'call-one', 'call-one', 'call-two', 'call-two', 'call-one', 'call-one'
    ])

    expect(ids).toEqual(Array(6).fill(expect.any(String)))
    expect(new Set(ids).size).toBe(4)
    expect(ids.slice(4)).toEqual(ids.slice(0, 2))
  })
})
