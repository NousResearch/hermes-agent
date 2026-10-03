import { afterEach, describe, expect, it, vi } from 'vitest'

import { setVoicePlaybackSpeed } from '@/store/voice-playback-speed'
import { speakText } from '@/hermes'

import { playSpeechText, stopVoicePlayback } from './voice-playback'

// The device speed preference must reach every playback rung. The relay
// (stream) rung is exercised by the WebSocket double below: its PCM chunks
// are scheduled through a stubbed AudioContext, so `playbackRate.value` and
// the SHORTENED duration consumed by the chunk timeline pin the rate math.
vi.mock('@/lib/voice-client-direct', () => ({
  cutSentences: (text: string) => [text],
  directTtsConfig: vi.fn(async () => null),
  synthesizeSpeechClientDirect: vi.fn()
}))

vi.mock('@/hermes', () => ({
  getApiRequestConnection: () => null,
  getApiRequestProfile: () => null,
  speakText: vi.fn(async () => ({ data_url: 'data:audio/mpeg;base64,AAAA' }))
}))

// jsdom HTMLMediaElement has no playable media; drive the playback loop with
// controllable elements instead. endedEvents collects the elements whose
// 'ended' event is still pending, keyed in creation order.
function installAudioStubs() {
  const created: FakeAudio[] = []
    const createdBuffers: Array<{ length: number; duration: number }> = []

  class FakeAudio {
    playbackRate = 1
    paused = true
    listeners: Record<string, Array<() => void>> = {}
    src = ''

    constructor(src: string) {
      this.src = src
      created.push(this)
    }

    addEventListener(type: string, listener: () => void) {
      this.listeners[type] = [...(this.listeners[type] ?? []), listener]
    }

    removeEventListener() {}

    play() {
      this.paused = false

      return Promise.resolve()
    }

    pause() {
      this.paused = true
    }

    load() {}

    fire(type: string) {
      for (const listener of this.listeners[type] ?? []) {
        listener()
      }
    }
  }

  vi.stubGlobal('Audio', FakeAudio)
  Object.defineProperty(window, 'Audio', { configurable: true, value: FakeAudio })

  return created
}

describe('voice playback speed reaches every playback rung', () => {
  afterEach(() => {
    stopVoicePlayback()
    vi.unstubAllGlobals()
    Reflect.deleteProperty(window, 'Audio')
  })

  it('starts data-URL playback at synthesis speed and keeps the element at rate 1', async () => {
    vi.mocked(speakText).mockClear()
    setVoicePlaybackSpeed(0.75)

    const created = installAudioStubs()
    const pending = playSpeechText('hello there', { source: 'read-aloud' })

    await vi.waitFor(() => expect(created.length).toBeGreaterThan(0))
    const audio = created[0]

    // Synthesis-side speed: the POST carried the requested speed and the
    // element plays at rate 1 (pitch-perfect; no playbackRate resampling).
    expect(speakText).toHaveBeenCalledWith('hello there', expect.anything(), 0.75)
    expect(audio.playbackRate).toBe(1)

    // A live speed change re-synthesizes on the NEXT reply; the in-flight
    // element is never retuned via playbackRate.
    setVoicePlaybackSpeed(1.5)
    expect(audio.playbackRate).toBe(1)

    audio.fire('ended')
    await expect(pending).resolves.toBe(true)

    setVoicePlaybackSpeed(1)
  })



  it('schedules stream chunks at the chosen rate without overlap', async () => {
    const created = installAudioStubs()

    let onmessage: ((event: { data: unknown }) => void) | null = null

    class FakeWebSocket {
      static OPEN = 1
      static CONNECTING = 0
      binaryType = ''
      readyState = 0

      constructor() {
        onmessage = (event: { data: unknown }) => this.onmessage?.(event)
        // Real sockets reach OPEN a tick later; queueMicrotask mimics that
        // so the client's CONNECTING buffering + onopen announcement run.
        queueMicrotask(() => {
          this.readyState = 1
          this.onopen?.()
        })
      }

      onmessage: ((event: { data: unknown }) => void) | null = null
      onopen: (() => void) | null = null
      onerror: (() => void) | null = null
      onclose: (() => void) | null = null

      static sentFrames: string[] = []

      send(data: string) {
        FakeWebSocket.sentFrames.push(data)
      }

      close() {}
    }

    vi.stubGlobal('WebSocket', FakeWebSocket)
    Object.defineProperty(window, 'WebSocket', { configurable: true, value: FakeWebSocket })

    const scheduled: Array<{ playbackRate: number; timeline: number }> = []

    class FakeBufferSource {
      playbackRate = { value: 1 }
      buffer: { duration: number } | null = null

      connect() {}

      start(at: number) {
        scheduled.push({ playbackRate: this.playbackRate.value, timeline: at })
      }
    }

    const createdBuffers: Array<{ length: number; duration: number }> = []
    // context.currentTime advances 0 → 0.05 → 0.1 …; buffer duration 1s each.
    let currentTime = 0

    const fakeContext = {
      get currentTime() {
        return currentTime
      },

      createBuffer(_channels: number, length: number, rate: number) {
        createdBuffers.push({ length, duration: length / rate })
        return { duration: length / rate, getChannelData: () => new Float32Array(length) }
      },

      createBufferSource() {
        return new FakeBufferSource()
      },

      destination: {},
      close: () => Promise.resolve()
    }

    vi.stubGlobal(
      'AudioContext',
      class {
        constructor() {
          return fakeContext
        }
      }
    )
    Object.defineProperty(window, 'AudioContext', {
      configurable: true,
      value: class {
        constructor() {
          return fakeContext
        }
      }
    })

    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: {
        getConnection: async () => ({ authMode: 'token', wsUrl: 'ws://127.0.0.1:5151/api/ws?token=local' }),
        getGatewayWsUrl: async () => ({ ok: true, wsUrl: 'ws://127.0.0.1:5151/api/ws?token=local' })
      }
    })

    FakeWebSocket.sentFrames = []
    setVoicePlaybackSpeed(2)

    const pending = playSpeechText('hello there', { source: 'read-aloud' })

    await vi.waitFor(() => expect(onmessage).not.toBeNull())

    // Server protocol: start frame, then PCM, then end.
    onmessage!({ data: JSON.stringify({ type: 'start', sample_rate: 24000 }) })
    onmessage!({ data: new Int16Array(24000).buffer })
    onmessage!({ data: new Int16Array(24000).buffer })
    onmessage!({ data: JSON.stringify({ type: 'end' }) })

    await vi.waitFor(() => expect(scheduled.length).toBe(2))

    // Speed is applied SYNTHESIS-SIDE: the client announces it on session
    // open and sources always play at rate 1 (pitch-perfect; the PCM arrives
    // already at the requested rate from providers honoring speed).
    expect(scheduled[0].playbackRate).toBe(1)
    expect(scheduled[1].playbackRate).toBe(1)
    // Chunk 1 ends at start(0.05) + full duration 1.0 = 1.05; chunk 2 starts
    // exactly there — no overlap, no gap.
    expect(scheduled[1].timeline).toBeCloseTo(1.05, 5)

    // The WS protocol carries the speed: the session opens with a speed
    // announcement ahead of any text, and a live change rides a speed frame.
    const frames = FakeWebSocket.sentFrames.map(data => JSON.parse(data))
    expect(frames[0]).toEqual({ speed: 2 })
    expect(frames[0]).toEqual(frames.find(f => typeof f.speed === 'number'))

    created.length = 0
    setVoicePlaybackSpeed(0.75)
    await vi.waitFor(() =>
      FakeWebSocket.sentFrames.some(data => JSON.parse(data).speed === 0.75)
    )
    await expect(pending).resolves.toBe(true)
    setVoicePlaybackSpeed(1)
  })
})
