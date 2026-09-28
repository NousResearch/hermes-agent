import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { directTtsConfig, synthesizeSpeechClientDirect } from '@/lib/voice-client-direct'

import { startSpeechStream, stopVoicePlayback } from './voice-playback'

// The live speech queue must never outlive what the user can hear. Two ways it
// did: a suspended AudioContext froze the clock the stream rung derived its
// `done` from, so the whole buffered reply waited (and later played all at
// once), and the client-direct rung reported a mid-reply play() rejection as
// 'done', silently dropping the rest of the reply.

vi.mock('@/lib/voice-client-direct', () => ({
  directTtsConfig: vi.fn(async () => null),
  synthesizeSpeechClientDirect: vi.fn()
}))

vi.mock('@/hermes', () => ({
  getApiRequestConnection: () => null,
  getApiRequestProfile: () => null,
  speakText: vi.fn(async () => {
    throw new Error('no audio in test')
  })
}))

interface FakeSource {
  buffer: unknown
  connect: () => void
  onended: null | (() => void)
  start: ReturnType<typeof vi.fn>
  stop: ReturnType<typeof vi.fn>
}

class FakeAudioContext {
  static instances: FakeAudioContext[] = []
  static initialState: AudioContextState = 'running'
  state: AudioContextState = FakeAudioContext.initialState
  currentTime = 0
  destination = {}
  sources: FakeSource[] = []
  // Autoplay-denied Chromium leaves resume() pending rather than rejecting.
  resume = vi.fn(() => (this.state === 'running' ? Promise.resolve() : new Promise<void>(() => undefined)))
  close = vi.fn(async () => {
    this.state = 'closed'
  })

  constructor() {
    FakeAudioContext.instances.push(this)
  }

  createBuffer(_channels: number, length: number, rate: number) {
    return { duration: length / rate, getChannelData: () => new Float32Array(length) }
  }

  createBufferSource(): FakeSource {
    const source: FakeSource = { buffer: null, connect: () => undefined, onended: null, start: vi.fn(), stop: vi.fn() }
    this.sources.push(source)

    return source
  }
}

class FakeWebSocket {
  static CONNECTING = 0
  static OPEN = 1
  static instances: FakeWebSocket[] = []
  binaryType = ''
  readyState = FakeWebSocket.OPEN
  sent: string[] = []
  onopen: null | (() => void) = null
  onmessage: null | ((event: { data: unknown }) => void) = null
  onerror: null | (() => void) = null
  onclose: null | (() => void) = null

  constructor(readonly url: string) {
    FakeWebSocket.instances.push(this)
  }

  send(data: string) {
    this.sent.push(data)
  }

  close() {
    this.readyState = 3
  }

  receive(data: unknown) {
    this.onmessage?.({ data })
  }
}

/** Sources that were scheduled and never stopped — audio still on the timeline. */
function liveSources(context: FakeAudioContext) {
  return context.sources.filter(source => source.start.mock.calls.length > 0 && source.stop.mock.calls.length === 0)
}

/** One second of 24 kHz int16 PCM. */
const SECOND_OF_PCM = () => new Int16Array(24_000).buffer

async function openStream() {
  const session = await startSpeechStream({ source: 'voice-conversation' })

  if (!session) {
    throw new Error('expected the stream rung')
  }

  const outcome = vi.fn()
  void session.done.then(outcome)
  const ws = FakeWebSocket.instances.at(-1)!

  ws.receive(JSON.stringify({ sample_rate: 24_000, type: 'start' }))

  return { context: FakeAudioContext.instances.at(-1)!, outcome, session, ws }
}

beforeEach(() => {
  vi.useFakeTimers()
  FakeAudioContext.instances = []
  FakeAudioContext.initialState = 'running'
  FakeWebSocket.instances = []
  vi.stubGlobal('AudioContext', FakeAudioContext)
  vi.stubGlobal('WebSocket', FakeWebSocket)
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: {
      getConnection: async () => ({
        authMode: 'token',
        baseUrl: 'http://127.0.0.1:5151',
        wsUrl: 'ws://127.0.0.1:5151/api/ws?token=t'
      }),
      getGatewayWsUrl: async () => ({ ok: true, wsUrl: 'ws://127.0.0.1:5151/api/ws?token=t' })
    }
  })
})

afterEach(() => {
  stopVoicePlayback()
  Reflect.deleteProperty(window, 'hermesDesktop')
  vi.mocked(directTtsConfig).mockReset()
  vi.mocked(directTtsConfig).mockResolvedValue(null)
  vi.mocked(synthesizeSpeechClientDirect).mockReset()
  vi.useRealTimers()
  vi.unstubAllGlobals()
})

describe('speak-stream rung — settles from the audio clock, not the timeline length', () => {
  it('settles a suspended context with a frozen clock in bounded time and drops its backlog', async () => {
    FakeAudioContext.initialState = 'suspended'
    const { context, outcome, ws } = await openStream()

    // A fast provider: 20 s of reply buffered ahead on a clock that never moves.
    for (let chunk = 0; chunk < 20; chunk += 1) {
      ws.receive(SECOND_OF_PCM())
    }

    ws.receive(JSON.stringify({ type: 'end' }))
    expect(liveSources(context)).toHaveLength(20)

    await vi.advanceTimersByTimeAsync(3_000)

    // Nothing was ever audible → the caller speaks the reply another way,
    // and none of the 20 s can play later when something resumes audio.
    expect(outcome).toHaveBeenCalledWith('fallback')
    expect(liveSources(context)).toHaveLength(0)
    expect(context.close).toHaveBeenCalled()
  })

  it('retries the resume before giving up on a suspended context', async () => {
    FakeAudioContext.initialState = 'suspended'
    const { context } = await openStream()

    await vi.advanceTimersByTimeAsync(3_000)

    expect(context.resume.mock.calls.length).toBeGreaterThanOrEqual(2)
  })

  it('plays a running stream to its end instead of cutting it off', async () => {
    const { context, outcome, ws } = await openStream()

    ws.receive(SECOND_OF_PCM())
    ws.receive(SECOND_OF_PCM())
    ws.receive(JSON.stringify({ type: 'end' }))

    // The clock advances in real time: 1.5 s in, audio is still playing.
    for (let step = 0; step < 15; step += 1) {
      context.currentTime += 0.1
      await vi.advanceTimersByTimeAsync(100)
    }

    expect(outcome).not.toHaveBeenCalled()

    for (let step = 0; step < 7; step += 1) {
      context.currentTime += 0.1
      await vi.advanceTimersByTimeAsync(100)
    }

    expect(outcome).toHaveBeenCalledWith('done')
  })

  it('reports done once audio was heard, even if the clock stalls afterwards', async () => {
    const { context, outcome, ws } = await openStream()

    ws.receive(SECOND_OF_PCM())
    ws.receive(SECOND_OF_PCM())
    context.currentTime = 0.5
    await vi.advanceTimersByTimeAsync(100)

    // Device loss mid-reply: the clock freezes.
    await vi.advanceTimersByTimeAsync(3_000)

    expect(outcome).toHaveBeenCalledWith('done')
    expect(liveSources(context)).toHaveLength(0)
  })

  it('stopVoicePlayback leaves zero scheduled sources', async () => {
    const { context, outcome, ws } = await openStream()

    for (let chunk = 0; chunk < 5; chunk += 1) {
      ws.receive(SECOND_OF_PCM())
    }

    stopVoicePlayback()
    await vi.advanceTimersByTimeAsync(0)

    expect(outcome).toHaveBeenCalledWith('done')
    expect(liveSources(context)).toHaveLength(0)
    expect(context.close).toHaveBeenCalled()
  })
})

describe('client-direct rung — a mid-reply failure never drops the rest as done', () => {
  class FakeAudio extends EventTarget {
    static instances: FakeAudio[] = []
    /** Per-clip play() behavior, by creation order. */
    static plan: ('ok' | 'reject')[] = []
    play = vi.fn(() => {
      if (FakeAudio.plan[FakeAudio.instances.indexOf(this)] === 'reject') {
        return Promise.reject(new DOMException('no gesture', 'NotAllowedError'))
      }

      window.setTimeout(() => this.dispatchEvent(new Event('ended')), 10)

      return Promise.resolve()
    })

    pause = vi.fn()
    src: string

    constructor(src: string) {
      super()
      this.src = src
      FakeAudio.instances.push(this)
    }
  }

  const tts = {
    api_key: 'xi_test',
    base_url: 'https://api.elevenlabs.io/v1',
    min_len: 1,
    mode: 'direct',
    model: 'eleven_flash_v2_5',
    provider: 'elevenlabs',
    speed: null,
    voice: 'voice-1',
    wire: 'elevenlabs-tts'
  } as const

  beforeEach(() => {
    FakeAudio.instances = []
    FakeAudio.plan = []
    vi.stubGlobal('Audio', FakeAudio)
    vi.stubGlobal('URL', Object.assign(URL, { createObjectURL: () => 'blob:clip', revokeObjectURL: () => undefined }))
    vi.mocked(directTtsConfig).mockResolvedValue(tts as never)
    vi.mocked(synthesizeSpeechClientDirect).mockImplementation(async () => new Uint8Array([1, 2, 3]).buffer)
  })

  async function speak(text: string) {
    const session = await startSpeechStream({ source: 'voice-conversation' })

    if (!session) {
      throw new Error('expected the client-direct rung')
    }

    const outcome = vi.fn()
    void session.done.then(outcome)
    session.append(text)
    session.finish()
    await vi.advanceTimersByTimeAsync(5_000)

    return outcome
  }

  it('plays every sentence and reports done on the happy path', async () => {
    FakeAudio.plan = ['ok', 'ok']

    const outcome = await speak('The first sentence plays. The second sentence plays too.')

    expect(FakeAudio.instances).toHaveLength(2)
    expect(outcome).toHaveBeenCalledWith('done')
  })

  it('sentence 1 plays, sentence 2 play() rejects → fallback, after one unlock retry', async () => {
    FakeAudio.plan = ['ok', 'reject']

    const outcome = await speak('The first sentence plays. The second sentence is blocked.')

    expect(FakeAudio.instances[1].play).toHaveBeenCalledTimes(2)
    expect(outcome).toHaveBeenCalledWith('fallback')
    expect(outcome).not.toHaveBeenCalledWith('done')
  })

  it('a provider failure after sentence 1 is fallback too, not a silent drop', async () => {
    FakeAudio.plan = ['ok', 'ok']
    vi.mocked(synthesizeSpeechClientDirect)
      .mockImplementationOnce(async () => new Uint8Array([1]).buffer)
      .mockRejectedValueOnce(new Error('ElevenLabs TTS error (HTTP 429)'))

    const outcome = await speak('The first sentence plays. The second sentence is rate limited.')

    expect(outcome).toHaveBeenCalledWith('fallback')
  })

  it('cancels an in-flight synthesis on barge-in', async () => {
    let signal: AbortSignal | undefined

    vi.mocked(synthesizeSpeechClientDirect).mockImplementation(
      (_tts, _text, abort) =>
        new Promise<ArrayBuffer>(() => {
          signal = abort
        })
    )

    const session = await startSpeechStream({ source: 'voice-conversation' })
    session!.append('A sentence that never finishes synthesizing.')
    session!.finish()
    await vi.advanceTimersByTimeAsync(0)

    stopVoicePlayback()

    expect(signal?.aborted).toBe(true)
  })
})
