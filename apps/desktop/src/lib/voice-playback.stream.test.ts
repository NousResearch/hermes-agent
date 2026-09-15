import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $voicePlayback } from '@/store/voice-playback'

import { directTtsConfig } from './voice-client-direct'
import {
  openSpeechStream,
  SPEECH_STREAM_FIRST_AUDIO_TIMEOUT_MS,
  startSpeechStream,
  stopVoicePlayback
} from './voice-playback'

vi.mock('./voice-client-direct', () => ({
  directTtsConfig: vi.fn(async () => null),
  cutSentences: vi.fn(),
  synthesizeSpeechClientDirect: vi.fn()
}))

class FakeAudioContext {
  static instances: FakeAudioContext[] = []
  starts: number[] = []
  currentTime = 0
  destination = {}
  state: AudioContextState = 'running'

  close = vi.fn(async () => undefined)
  resume = vi.fn(async () => undefined)

  constructor() {
    FakeAudioContext.instances.push(this)
  }

  createBuffer(_channels: number, length: number, rate: number) {
    return { duration: length / rate, getChannelData: () => new Float32Array(length) }
  }

  createBufferSource() {
    return { buffer: null, connect: vi.fn(), start: (at: number) => this.starts.push(at) }
  }
}

class FakeWebSocket {
  static readonly CONNECTING = 0
  static readonly OPEN = 1
  static readonly CLOSED = 3
  static instances: FakeWebSocket[] = []

  binaryType = ''
  readyState = FakeWebSocket.OPEN
  sent: string[] = []
  onclose: (() => void) | null = null
  onerror: (() => void) | null = null
  onmessage: ((event: { data: ArrayBuffer | string }) => void) | null = null
  onopen: (() => void) | null = null

  constructor(readonly url: string) {
    FakeWebSocket.instances.push(this)
  }

  close() {
    this.readyState = FakeWebSocket.CLOSED
  }

  send(data: string) {
    this.sent.push(data)
  }

  emitJson(frame: object) {
    this.onmessage?.({ data: JSON.stringify(frame) })
  }
}

describe('openSpeechStream no-audio fallback', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    FakeWebSocket.instances = []
    FakeAudioContext.instances = []
    vi.stubGlobal('WebSocket', FakeWebSocket)
    vi.stubGlobal('AudioContext', FakeAudioContext)
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: {
        getConnection: vi.fn(async () => ({ authMode: 'token', wsUrl: 'ws://127.0.0.1/api/ws?token=local' })),
        getGatewayWsUrl: vi.fn(async () => ({ ok: true, wsUrl: 'ws://127.0.0.1/api/ws?token=local' }))
      }
    })
  })

  afterEach(() => {
    stopVoicePlayback()
    Reflect.deleteProperty(window, 'hermesDesktop')
    vi.clearAllMocks()
    vi.useRealTimers()
    vi.unstubAllGlobals()
  })

  it('plays PCM chunks in arrival order without overlapping their scheduled times', async () => {
    const session = openSpeechStream('ws://127.0.0.1/speak-stream', { source: 'voice-conversation' })
    const socket = FakeWebSocket.instances[0]
    socket.emitJson({ type: 'start', sample_rate: 24_000 })
    socket.onmessage?.({ data: new Int16Array(2400).buffer })
    socket.onmessage?.({ data: new Int16Array(4800).buffer })
    expect(FakeAudioContext.instances[0].starts[0]).toBeCloseTo(0.05)
    expect(FakeAudioContext.instances[0].starts[1]).toBeCloseTo(0.15)
    session.cancel()
    await expect(session.done).resolves.toBe('done')
  })

  it('does not reopen audio when stopped during async config lookup', async () => {
    let resolveConfig: (value: null) => void = () => undefined
    vi.mocked(directTtsConfig).mockImplementationOnce(
      () =>
        new Promise(resolve => {
          resolveConfig = resolve
        })
    )
    const pending = startSpeechStream({ source: 'read-aloud' })
    stopVoicePlayback()
    resolveConfig(null)
    await expect(pending).resolves.toBeNull()
    expect(FakeWebSocket.instances).toHaveLength(0)
    expect($voicePlayback.get().status).toBe('idle')
  })

  it('an old completion cannot reset or cancel a newer preparing session', async () => {
    const old = await startSpeechStream({ source: 'read-aloud' })
    expect(old).not.toBeNull()
    const next = await startSpeechStream({ source: 'voice-conversation' })
    await Promise.resolve()
    expect($voicePlayback.get().source).toBe('voice-conversation')
    expect($voicePlayback.get().status).toBe('preparing')
    old?.cancel()
    stopVoicePlayback()
    expect(FakeWebSocket.instances[1].readyState).toBe(FakeWebSocket.CLOSED)
    await expect(next?.done).resolves.toBe('done')
  })

  it('falls back when accepted text produces no PCM before the deadline', async () => {
    const session = openSpeechStream('ws://127.0.0.1/speak-stream', {
      source: 'voice-conversation'
    })

    session.append('一文目です。')
    await vi.advanceTimersByTimeAsync(SPEECH_STREAM_FIRST_AUDIO_TIMEOUT_MS)

    await expect(session.done).resolves.toBe('fallback')
  })

  it('treats an end frame before the first PCM frame as fallback', async () => {
    const session = openSpeechStream('ws://127.0.0.1/speak-stream', {
      source: 'voice-conversation'
    })

    const socket = FakeWebSocket.instances[0]

    session.append('一文目です。')
    socket.emitJson({ type: 'start', sample_rate: 24_000, channels: 1 })
    socket.emitJson({ type: 'end' })

    await expect(session.done).resolves.toBe('fallback')
  })
})
