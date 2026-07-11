// @vitest-environment jsdom
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { RealtimeVoiceSession, reduceRealtimeVoiceEvent } from './realtime-voice-session'

class FakeTrack {
  enabled = true
  stop = vi.fn()
}

class FakeStream {
  readonly track = new FakeTrack()

  getAudioTracks() {
    return [this.track] as unknown as MediaStreamTrack[]
  }

  getTracks() {
    return this.getAudioTracks()
  }
}

class FakeDataChannel extends EventTarget {
  readyState: RTCDataChannelState = 'open'
  readonly sent: string[] = []
  close = vi.fn(() => {
    this.readyState = 'closed'
  })

  send(value: string) {
    this.sent.push(value)
  }

  emit(event: unknown) {
    this.dispatchEvent(new MessageEvent('message', { data: JSON.stringify(event) }))
  }
}

class FakePeer extends EventTarget {
  readonly channel = new FakeDataChannel()
  connectionState: RTCPeerConnectionState = 'new'
  close = vi.fn(() => {
    this.connectionState = 'closed'
  })
  addTrack = vi.fn()
  createDataChannel = vi.fn(() => this.channel as unknown as RTCDataChannel)
  createOffer = vi.fn(async () => ({ sdp: 'offer-sdp', type: 'offer' }) as RTCSessionDescriptionInit)
  setLocalDescription = vi.fn(async () => undefined)
  setRemoteDescription = vi.fn(async () => undefined)

  connect() {
    this.connectionState = 'connected'
    this.dispatchEvent(new Event('connectionstatechange'))
  }
}

function fixture() {
  const peers: FakePeer[] = []
  const streams: FakeStream[] = []
  const transcripts = vi.fn()
  const onError = vi.fn()
  const onSpeechStarted = vi.fn()
  const onStatus = vi.fn()
  const fetchImpl = vi.fn(async () => new Response('answer-sdp', { status: 200 }))

  const createSession = vi.fn(async () => ({
    client_secret: 'ek_short_lived_renderer_secret',
    expires_at: Math.floor(Date.now() / 1000) + 60,
    ok: true,
    session_binding: 'binding-1'
  }))

  const session = new RealtimeVoiceSession(
    { onError, onSpeechStarted, onStatus, onTranscript: transcripts },
    {
      createPeerConnection: () => {
        const peer = new FakePeer()
        peers.push(peer)

        return peer as unknown as RTCPeerConnection
      },
      createSession,
      fetch: fetchImpl as unknown as typeof fetch,
      getUserMedia: async () => {
        const stream = new FakeStream()
        streams.push(stream)

        return stream as unknown as MediaStream
      }
    }
  )

  return { createSession, fetchImpl, onError, onSpeechStarted, onStatus, peers, session, streams, transcripts }
}

describe('RealtimeVoiceSession', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
  })

  it('negotiates WebRTC with only the ephemeral secret and never persists it', async () => {
    const setItem = vi.spyOn(Storage.prototype, 'setItem')
    const { createSession, fetchImpl, peers, session } = fixture()

    await session.connect({ language: 'ko', sessionId: 'desktop-session-1' })

    expect(createSession).toHaveBeenCalledWith('desktop-session-1', 'ko')
    expect(fetchImpl).toHaveBeenCalledWith(
      'https://api.openai.com/v1/realtime/calls',
      expect.objectContaining({
        body: 'offer-sdp',
        headers: {
          Authorization: 'Bearer ek_short_lived_renderer_secret',
          'Content-Type': 'application/sdp'
        },
        method: 'POST'
      })
    )
    expect(peers[0]?.setRemoteDescription).toHaveBeenCalledWith({ type: 'answer', sdp: 'answer-sdp' })
    expect(setItem).not.toHaveBeenCalled()
  })

  it('accepts ten final transcripts exactly once and emits no response-generation event', async () => {
    const { peers, session, transcripts } = fixture()

    await session.connect({ sessionId: 'desktop-session-10-turn' })
    const channel = peers[0]!.channel

    for (let turn = 1; turn <= 10; turn += 1) {
      const event = {
        type: 'conversation.item.input_audio_transcription.completed',
        item_id: `item-${turn}`,
        transcript: `한국어 발화 ${turn}`
      }

      channel.emit(event)
      channel.emit(event)
    }

    session.cancelInput()

    expect(transcripts).toHaveBeenCalledTimes(10)
    expect(transcripts.mock.calls.map(([turn]) => turn.id)).toEqual(
      Array.from({ length: 10 }, (_, index) => `item-${index + 1}`)
    )
    expect(channel.sent.map(value => JSON.parse(value))).toEqual([{ type: 'input_audio_buffer.clear' }])
    expect(channel.sent.some(value => value.includes('response.'))).toBe(false)
    expect(channel.sent.some(value => value.includes('function_call'))).toBe(false)
  })

  it('reports VAD state, supports mute, and fully cleans up disconnect/reconnect', async () => {
    const { onSpeechStarted, onStatus, peers, session, streams } = fixture()

    await session.connect({ sessionId: 'desktop-session-cleanup' })
    peers[0]!.connect()
    peers[0]!.channel.emit({ type: 'input_audio_buffer.speech_started', item_id: 'item-a' })
    peers[0]!.channel.emit({ type: 'input_audio_buffer.speech_stopped', item_id: 'item-a' })
    session.setMuted(true)

    expect(onSpeechStarted).toHaveBeenCalledTimes(1)
    expect(onStatus).toHaveBeenCalledWith('listening')
    expect(onStatus).toHaveBeenCalledWith('transcribing')
    expect(streams[0]!.track.enabled).toBe(false)

    await session.connect({ sessionId: 'desktop-session-reconnect' })

    expect(peers[0]!.channel.close).toHaveBeenCalledTimes(1)
    expect(peers[0]!.close).toHaveBeenCalledTimes(1)
    expect(streams[0]!.track.stop).toHaveBeenCalledTimes(1)

    session.disconnect()
    expect(peers[1]!.channel.close).toHaveBeenCalledTimes(1)
    expect(peers[1]!.close).toHaveBeenCalledTimes(1)
    expect(streams[1]!.track.stop).toHaveBeenCalledTimes(1)
    expect(session.currentStatus).toBe('idle')
  })

  it('fails closed on beta response audio events and cleans resources', async () => {
    const { onError, peers, session, streams } = fixture()

    await session.connect({ sessionId: 'desktop-session-invalid-event' })
    peers[0]!.channel.emit({ type: 'response.audio.delta', delta: 'forbidden' })

    expect(onError).toHaveBeenCalledWith(
      expect.objectContaining({ message: expect.stringContaining('response-generation') })
    )
    expect(peers[0]!.close).toHaveBeenCalledTimes(1)
    expect(streams[0]!.track.stop).toHaveBeenCalledTimes(1)
    expect(session.currentStatus).toBe('error')
  })
})

describe('reduceRealtimeVoiceEvent', () => {
  it('recognizes only GA transcription/VAD events and rejects response generation', () => {
    expect(
      reduceRealtimeVoiceEvent({
        type: 'conversation.item.input_audio_transcription.completed',
        item_id: 'item-1',
        transcript: '  hello  '
      })
    ).toEqual({ status: 'listening', transcript: { id: 'item-1', text: 'hello' } })
    expect(reduceRealtimeVoiceEvent({ type: 'input_audio_buffer.speech_started' })).toEqual({
      speechStarted: true,
      status: 'listening'
    })
    expect(reduceRealtimeVoiceEvent({ type: 'response.create' })).toEqual({
      error: 'Unexpected Realtime response-generation event',
      status: 'error'
    })
    expect(reduceRealtimeVoiceEvent({ type: 'response.audio.delta' })).toEqual({
      error: 'Unexpected Realtime response-generation event',
      status: 'error'
    })
  })
})
