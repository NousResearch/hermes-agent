/**
 * Tests for the shared realtime voice client's protocol logic.
 *
 * Audio is disabled (test seam); the socket is a recording fake. Covers the
 * supervisor contract that must stay mirrored with agent/voice_supervisor.py:
 * function-call dispatch, silent-tool-call detection, deferred
 * response.create after tool output, force_message verbatim speech, and
 * barge-in playback clearing.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  ACK_PHRASES,
  type RealtimeFunctionCall,
  RealtimeVoiceClient,
  type RealtimeVoiceStatus,
  resampleFloat32
} from './realtime-voice'

const WS_CONNECTING = 0
const WS_OPEN = 1
const WS_CLOSED = 3

class FakeSocket {
  static instances: FakeSocket[] = []
  readonly sent: Record<string, unknown>[] = []
  readyState = WS_CONNECTING
  onopen: (() => void) | null = null
  onerror: (() => void) | null = null
  onclose: ((ev: { reason: string }) => void) | null = null
  onmessage: ((ev: { data: string }) => void) | null = null

  readonly url: string
  readonly protocols: string[]

  constructor(url: string, protocols: string[]) {
    this.url = url
    this.protocols = protocols
    FakeSocket.instances.push(this)
  }

  open(): void {
    this.readyState = WS_OPEN
    this.onopen?.()
  }

  send(payload: string): void {
    this.sent.push(JSON.parse(payload) as Record<string, unknown>)
  }

  close(): void {
    this.readyState = WS_CLOSED
  }

  sentTypes(): string[] {
    return this.sent.map(f => String(f.type))
  }
}

interface Harness {
  client: RealtimeVoiceClient
  socket: FakeSocket
  functionCalls: RealtimeFunctionCall[]
  statuses: RealtimeVoiceStatus[]
  userTranscripts: string[]
  speechStarts: number[]
}

async function connect(): Promise<Harness> {
  FakeSocket.instances = []

  const h: Omit<Harness, 'client' | 'socket'> = {
    functionCalls: [],
    statuses: [],
    userTranscripts: [],
    speechStarts: []
  }

  const client = new RealtimeVoiceClient({
    disableAudio: true,
    createSocket: (url, protocols) => new FakeSocket(url, protocols) as unknown as WebSocket
  })

  const pending = client.connect(
    {
      token: 'eph-1',
      url: 'wss://api.x.ai/v1/realtime?model=grok-voice-latest',
      session_update: { type: 'session.update', session: { marker: true } }
    },
    {
      onFunctionCall: call => {
        h.functionCalls.push(call)
      },
      onStatus: status => h.statuses.push(status),
      onUserTranscript: text => h.userTranscripts.push(text),
      onUserSpeechStarted: () => {
        h.speechStarts.push(Date.now())
      }
    }
  )

  const socket = FakeSocket.instances[0]
  socket.open()
  await pending

  return { ...h, client, socket }
}

beforeEach(() => {
  vi.useFakeTimers()
})

afterEach(() => {
  vi.useRealTimers()
})

describe('connect', () => {
  it('close() during the mic permission prompt releases the mic and never dials', async () => {
    FakeSocket.instances = []
    const track = { stop: vi.fn() }
    let grantMic: (stream: unknown) => void = () => undefined

    const getUserMedia = vi.fn(
      () =>
        new Promise(resolve => {
          grantMic = resolve
        })
    )

    class FakeAudioContext {
      sampleRate = 16000
      destination = {}
      close = vi.fn(async () => undefined)
      createMediaStreamSource() {
        return { connect: vi.fn() }
      }
      createScriptProcessor() {
        return { onaudioprocess: null, connect: vi.fn(), disconnect: vi.fn() }
      }
      createGain() {
        return { gain: { value: 1 }, connect: vi.fn() }
      }
    }

    vi.stubGlobal('navigator', { mediaDevices: { getUserMedia } })
    vi.stubGlobal('AudioContext', FakeAudioContext)

    try {
      const client = new RealtimeVoiceClient({
        createSocket: (url, protocols) => new FakeSocket(url, protocols) as unknown as WebSocket
      })

      const statuses: RealtimeVoiceStatus[] = []

      const pending = client.connect(
        {
          token: 'eph-1',
          url: 'wss://api.x.ai/v1/realtime?model=grok-voice-latest',
          session_update: { type: 'session.update', session: {} }
        },
        {
          onFunctionCall: () => undefined,
          onStatus: status => {
            statuses.push(status)
          }
        }
      )

      // User stops the call while the browser is still asking for the mic.
      client.close()
      grantMic({ getTracks: () => [track] })
      await pending

      expect(FakeSocket.instances).toHaveLength(0)
      expect(track.stop).toHaveBeenCalledTimes(1)
      expect(client.alive).toBe(false)
      expect(statuses).toEqual(['connecting', 'closed'])
    } finally {
      vi.unstubAllGlobals()
    }
  })

  it('authenticates via subprotocol and sends session.update verbatim', async () => {
    const h = await connect()
    expect(h.socket.protocols).toEqual(['xai-client-secret.eph-1'])
    expect(h.socket.sent[0]).toEqual({
      type: 'session.update',
      session: { marker: true }
    })
    expect(h.statuses).toEqual(['connecting', 'listening'])
  })

  it('handles server events that arrive before handshake resolves', async () => {
    FakeSocket.instances = []

    const client = new RealtimeVoiceClient({
      disableAudio: true,
      createSocket: (url, protocols) => new FakeSocket(url, protocols) as unknown as WebSocket
    })

    const pending = client.connect(
      {
        token: 'eph-1',
        url: 'wss://api.x.ai/v1/realtime?model=grok-voice-latest',
        session_update: { type: 'session.update', session: { marker: true } }
      },
      { onFunctionCall: () => undefined }
    )

    const socket = FakeSocket.instances[0]
    socket.onmessage?.({
      data: JSON.stringify({ type: 'response.created' })
    })
    socket.open()
    await pending
    client.sendFunctionOutput('c1', 'done')
    expect(socket.sentTypes()).not.toContain('response.create')
  })

  it('retries session.update without compat extras after a server error', async () => {
    const h = await connect()
    h.client.handleServerEvent({ type: 'error', error: 'unknown field reasoning' })
    const updates = h.socket.sent.filter(f => f.type === 'session.update')
    expect(updates).toHaveLength(2)
    expect((updates[1].session as { reasoning?: unknown }).reasoning).toBeUndefined()
  })
})

describe('supervisor tool contract', () => {
  it('dispatches function calls with parsed args', async () => {
    const h = await connect()
    h.client.handleServerEvent({
      type: 'response.function_call_arguments.done',
      name: 'consult_hermes',
      call_id: 'c1',
      arguments: JSON.stringify({ task: 'list files' })
    })
    expect(h.functionCalls).toEqual([{ name: 'consult_hermes', callId: 'c1', args: { task: 'list files' } }])
  })

  it('tracks whether the current response produced audio (silent-call detection)', async () => {
    const h = await connect()
    h.client.handleServerEvent({ type: 'response.created' })
    expect(h.client.lastResponseHadAudio).toBe(false)
    h.client.handleServerEvent({ type: 'response.output_audio.delta', delta: 'AAAA' })
    expect(h.client.lastResponseHadAudio).toBe(true)
    h.client.handleServerEvent({ type: 'response.created' })
    expect(h.client.lastResponseHadAudio).toBe(false)
  })

  it('sendFunctionOutput delivers the result then requests a response once quiet', async () => {
    const h = await connect()
    h.client.handleServerEvent({ type: 'response.created' })
    h.client.sendFunctionOutput('c1', 'done: 3 files')
    const item = h.socket.sent.find(f => f.type === 'conversation.item.create')
    expect(item?.item).toEqual({
      type: 'function_call_output',
      call_id: 'c1',
      output: 'done: 3 files'
    })
    // Response still active — the follow-up must wait.
    expect(h.socket.sentTypes()).not.toContain('response.create')
    h.client.handleServerEvent({ type: 'response.done' })
    await vi.advanceTimersByTimeAsync(400)
    expect(h.socket.sentTypes()).toContain('response.create')
  })

  it('speakVerbatim emits a force_message and acknowledgments rotate phrases', async () => {
    const h = await connect()
    h.client.speakVerbatim('Running tests.')

    const forced = h.socket.sent.find(
      f => f.type === 'conversation.item.create' && (f.item as { type?: string }).type === 'force_message'
    )

    expect((forced?.item as { content: { text: string }[] }).content[0].text).toBe('Running tests.')
    h.client.speakAcknowledgment()

    const acks = h.socket.sent.filter(
      f => f.type === 'conversation.item.create' && (f.item as { type?: string }).type === 'force_message'
    )

    const spoken = (acks[1].item as { content: { text: string }[] }).content[0].text
    expect(ACK_PHRASES).toContain(spoken)
  })
})

describe('resampleFloat32', () => {
  it('downsamples 48 kHz to 16 kHz without changing identity at equal rates', () => {
    const identity = new Float32Array([0, 0.5, -0.5, 1])
    expect(resampleFloat32(identity, 16000, 16000)).toBe(identity)

    // 3:1 decimation of a constant tone stays flat.
    const src = new Float32Array(4800).fill(0.25)
    const out = resampleFloat32(src, 48000, 16000)
    expect(out.length).toBe(1600)
    expect(out[0]).toBeCloseTo(0.25, 5)
    expect(out[out.length - 1]).toBeCloseTo(0.25, 5)
  })
})

describe('barge-in and transcripts', () => {
  it('speech_started clears playback state and notifies', async () => {
    const h = await connect()
    h.client.handleServerEvent({ type: 'input_audio_buffer.speech_started' })
    expect(h.speechStarts).toHaveLength(1)
    expect(h.client.speaking).toBe(false)
  })

  it('user transcripts surface for stop-word detection only', async () => {
    const h = await connect()
    h.client.handleServerEvent({
      type: 'conversation.item.input_audio_transcription.completed',
      transcript: 'stop please'
    })
    expect(h.userTranscripts).toEqual(['stop please'])
    // Assistant speech transcripts are intentionally ignored — no captions.
    h.client.handleServerEvent({
      type: 'response.output_audio_transcript.done',
      transcript: 'On it.'
    })
    expect(h.userTranscripts).toEqual(['stop please'])
  })
})
