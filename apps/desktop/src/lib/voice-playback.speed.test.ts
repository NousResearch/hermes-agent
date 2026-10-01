import { afterEach, describe, expect, it, vi } from 'vitest'

import { setVoicePlaybackSpeed } from '@/store/voice-playback-speed'

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

  it('starts data-URL playback at the chosen rate and retunes it live', async () => {
    const created = installAudioStubs()
    setVoicePlaybackSpeed(1.5)

    const pending = playSpeechText('hello there', { source: 'read-aloud' })

    await vi.waitFor(() => expect(created.length).toBeGreaterThan(0))
    const audio = created[0]

    // Started at the preference, not 1x.
    expect(audio.playbackRate).toBe(1.5)

    // The user changes speed mid-reply — the live element is retuned.
    setVoicePlaybackSpeed(0.75)
    expect(audio.playbackRate).toBe(0.75)

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
      readyState = 1

      constructor() {
        onmessage = (event: { data: unknown }) => this.onmessage?.(event)
      }

      onmessage: ((event: { data: unknown }) => void) | null = null
      onopen: (() => void) | null = null
      onerror: (() => void) | null = null
      onclose: (() => void) | null = null

      send() {}

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

    // context.currentTime advances 0 → 0.05 → 0.1 …; buffer duration 1s each.
    let currentTime = 0

    const fakeContext = {
      get currentTime() {
        return currentTime
      },

      createBuffer(_channels: number, length: number, _rate: number) {
        return { duration: 1, getChannelData: () => new Float32Array(length) }
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

    setVoicePlaybackSpeed(2)

    const pending = playSpeechText('hello there', { source: 'read-aloud' })

    await vi.waitFor(() => expect(onmessage).not.toBeNull())

    // Server protocol: start frame, then PCM, then end.
    onmessage!({ data: JSON.stringify({ type: 'start', sample_rate: 24000 }) })
    onmessage!({ data: new Int16Array(24000).buffer })
    onmessage!({ data: new Int16Array(24000).buffer })
    onmessage!({ data: JSON.stringify({ type: 'end' }) })

    await vi.waitFor(() => expect(scheduled.length).toBe(2))

    expect(scheduled[0].playbackRate).toBe(2)
    expect(scheduled[1].playbackRate).toBe(2)
    // Chunk 1 ends at start(0.05) + duration/speed = 0.55; chunk 2 starts
    // exactly there — no overlap, no gap.
    expect(scheduled[1].timeline).toBeCloseTo(0.55, 5)

    created.length = 0
    await expect(pending).resolves.toBe(true)
    setVoicePlaybackSpeed(1)
  })
})
