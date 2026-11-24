// Teardown contract for voice-playback (#91991): a barge-in (stopVoicePlayback)
// must drain EVERY live playback — streaming sessions (WebSocket + AudioContext)
// and data-URL audio elements — and a settled session must never resume or
// write stale state into a newer turn.

import type * as SharedTypes from '@hermes/shared'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $voice, $voicePlayback } from '@/store/voice-playback'

import { playSpeechText, startSpeechStream, stopVoicePlayback } from './voice-playback'
import { $voiceTimeline, getVoiceOutputLatencyMs, readVoiceLevel } from './voice-timeline'

const gateway = vi.hoisted(() => {
  let nextUrl: string | null = 'ws://gateway.test/api/ws'

  return {
    currentUrl: () => nextUrl,
    setNextUrl: (url: string | null) => {
      nextUrl = url
    }
  }
})

vi.mock('@hermes/shared', async importOriginal => {
  const actual = await importOriginal<typeof SharedTypes>()

  return {
    ...actual,
    resolveGatewayWsUrl: async () => gateway.currentUrl()
  }
})

const direct = vi.hoisted(() => ({ config: null as object | null }))

vi.mock('@/lib/voice-client-direct', () => ({
  directTtsConfig: vi.fn(async () => direct.config),
  synthesizeSpeechClientDirect: vi.fn(async () => new ArrayBuffer(4))
}))

vi.mock('@/hermes', () => ({
  getApiRequestProfile: () => null,
  getApiRequestConnection: () => null,
  speakText: vi.fn(async () => ({
    data_url: 'data:audio/mpeg;base64,AAAA',
    mime_type: 'audio/mpeg',
    ok: true
  }))
}))

class FakeSocket {
  static readonly OPEN = 1
  static readonly CONNECTING = 0
  static instances: FakeSocket[] = []

  binaryType = 'blob'
  readyState = FakeSocket.CONNECTING
  sent: string[] = []
  url: string

  close = vi.fn(() => {
    this.readyState = 3
  })
  send = vi.fn((data: string) => {
    this.sent.push(data)
  })

  onopen: (() => void) | null = null
  onmessage: ((event: { data: string | ArrayBuffer }) => void) | null = null
  onclose: (() => void) | null = null
  onerror: (() => void) | null = null

  constructor(url: string) {
    this.url = url
    FakeSocket.instances.push(this)
  }

  serverOpen() {
    this.readyState = FakeSocket.OPEN
    this.onopen?.()
  }

  serverFrame(frame: object) {
    this.onmessage?.({ data: JSON.stringify(frame) })
  }

  serverPcm(bytes: number) {
    this.onmessage?.({ data: new ArrayBuffer(bytes) })
  }

  serverClose() {
    this.onclose?.()
  }

  serverError() {
    this.onerror?.()
  }
}

class FakeAudioContext {
  static instances: FakeAudioContext[] = []

  baseLatency = 0
  closed = false
  currentTime = 0
  destination = {}
  outputLatency = 0.2
  state = 'running'

  close = vi.fn(() => {
    this.closed = true

    return Promise.resolve()
  })
  resume = vi.fn(() => Promise.resolve())
  createBuffer = vi.fn((_channels: number, length: number, rate: number) => ({
    duration: length / rate,
    getChannelData: () => new Float32Array(length)
  }))
  createAnalyser = vi.fn(() => ({
    connect: vi.fn(),
    fftSize: 512,
    getFloatTimeDomainData: (out: Float32Array) => out.fill(0.1)
  }))
  createBufferSource = vi.fn(() => ({
    buffer: null as unknown,
    connect: vi.fn(),
    start: vi.fn()
  }))

  constructor() {
    FakeAudioContext.instances.push(this)
  }
}

class FakeAudio {
  static instances: FakeAudio[] = []

  paused = false
  src: string

  addEventListener = vi.fn()
  load = vi.fn()
  pause = vi.fn(() => {
    this.paused = true
  })
  play = vi.fn(() => {
    this.paused = false

    return Promise.resolve()
  })
  removeEventListener = vi.fn()

  constructor(src: string) {
    this.src = src
    FakeAudio.instances.push(this)
  }
}

async function liveSession() {
  const session = await startSpeechStream({ source: 'voice-conversation' })

  if (!session) {
    throw new Error('expected a live speech stream')
  }

  return session
}

describe('voice-playback teardown (#91991)', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    FakeSocket.instances = []
    FakeAudioContext.instances = []
    FakeAudio.instances = []
    gateway.setNextUrl('ws://gateway.test/api/ws')
    vi.stubGlobal('WebSocket', FakeSocket)
    vi.stubGlobal('AudioContext', FakeAudioContext)
    vi.stubGlobal('Audio', FakeAudio)
    vi.stubGlobal('hermesDesktop', { getConnection: vi.fn(async () => ({ profile: 'test' })) })
    $voicePlayback.set({ audioElement: null, messageId: null, sequence: 0, source: null, status: 'idle' })
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    vi.useRealTimers()
  })

  it('abandons a stream start that crosses a stop instead of resuming playback', async () => {
    // Stream discovery (gateway connection) is in flight when the user barges.
    const pending = startSpeechStream({ source: 'voice-conversation' })

    stopVoicePlayback()

    const session = await pending

    expect(session).toBeNull()
    expect(FakeSocket.instances).toHaveLength(0)
    expect($voicePlayback.get().status).toBe('idle')
  })

  it('does not let a settled session reset the playback state of a newer turn', async () => {
    const first = await liveSession()
    const second = await liveSession()

    // Starting the second session settles the first (its socket closes), but
    // the settled session's completion callback must not stomp the state the
    // newer session just wrote.
    expect(FakeSocket.instances[0]?.close).toHaveBeenCalled()
    expect($voicePlayback.get().status).toBe('preparing')

    stopVoicePlayback()
  })

  it('a barge-in closes the socket and audio context of a live streaming session and settles it', async () => {
    const session = await liveSession()
    const ws = FakeSocket.instances[0]

    ws.serverOpen()
    ws.serverFrame({ type: 'start', sample_rate: 24_000 })
    ws.serverPcm(64)

    expect($voicePlayback.get().status).toBe('speaking')

    stopVoicePlayback()

    expect(ws.close).toHaveBeenCalled()
    expect(FakeAudioContext.instances[0]?.closed).toBe(true)
    expect($voicePlayback.get().status).toBe('idle')
    await expect(session.done).resolves.toBe('done')

    // A settled session can no longer feed text — nothing reaches the socket.
    const sentBefore = ws.sent.length

    session.append('stale text')
    session.finish()

    expect(ws.sent.length).toBe(sentBefore)
  })

  it('a barge-in during the drain window settles the session and the late drain is a no-op', async () => {
    const session = await liveSession()
    const ws = FakeSocket.instances[0]

    ws.serverOpen()
    ws.serverFrame({ type: 'start', sample_rate: 24_000 })
    ws.serverPcm(64)
    ws.serverFrame({ type: 'end' }) // drains, then settles — timer pending

    stopVoicePlayback()

    await expect(session.done).resolves.toBe('done')
    expect(ws.close).toHaveBeenCalled()

    // The pending drain timer fires later — it must not double-settle or
    // disturb anything that started since.
    await vi.runOnlyPendingTimersAsync()

    expect($voicePlayback.get().status).toBe('idle')
  })

  it('a barge-in cuts a data-url playback and clears the audio element', async () => {
    gateway.setNextUrl(null) // no streaming endpoint → data-url fallback

    const playback = playSpeechText('hello world', { messageId: 'm1', source: 'read-aloud' })

    await vi.advanceTimersByTimeAsync(0)

    expect($voicePlayback.get().status).toBe('speaking')
    expect(FakeAudio.instances).toHaveLength(1)

    stopVoicePlayback()

    expect(FakeAudio.instances[0]?.pause).toHaveBeenCalled()
    expect(FakeAudio.instances[0]?.src).toBe('')
    expect(FakeAudio.instances[0]?.load).toHaveBeenCalled()
    expect($voicePlayback.get().status).toBe('idle')
    await expect(playback).resolves.toBe(false)
  })

  it('one stop drains BOTH of two overlapping streaming sessions (#91991)', async () => {
    // The reported overlap: session A is live with an open AudioContext;
    // session B starts (its stopVoicePlayback drains A under the Set
    // registry — the old single currentStop slot left A's socket open and
    // its done unsettled). A final explicit stop must still tear down
    // everything that is live.
    const first = await liveSession()

    const firstWs = FakeSocket.instances[0]

    firstWs.serverOpen()
    firstWs.serverFrame({ type: 'start', sample_rate: 24_000 })
    firstWs.serverPcm(64)
    first.append('turn one')

    const second = await liveSession()

    const secondWs = FakeSocket.instances[1]

    secondWs.serverOpen()
    secondWs.serverFrame({ type: 'start', sample_rate: 24_000 })
    secondWs.serverPcm(64)
    second.append('turn two')

    // Starting B drained A: both its socket and its AudioContext are gone.
    await expect(first.done).resolves.toBe('done')
    expect(firstWs.close).toHaveBeenCalled()
    expect(FakeAudioContext.instances[0]?.closed).toBe(true)

    stopVoicePlayback()

    await expect(second.done).resolves.toBe('done')
    expect(secondWs.close).toHaveBeenCalled()
    expect(FakeAudioContext.instances[1]?.closed).toBe(true)
    expect($voicePlayback.get().status).toBe('idle')
  })

  describe('what plugins see', () => {
    async function streaming() {
      $voiceTimeline.set({ segments: [] })
      await liveSession()
      const ws = FakeSocket.instances.at(-1)!

      ws.serverOpen()
      ws.serverFrame({ type: 'start', sample_rate: 24_000 })

      return ws
    }

    it('times words to when the listener hears them, and reports loudness', async () => {
      const ws = await streaming()
      const now = performance.now()

      ws.serverFrame({ type: 'segment', text: 'Hi', units: [{ endMs: 200, kind: 'word', startMs: 0, text: 'Hi' }] })
      ws.serverPcm(64)

      // 50 ms scheduling lead-in + 200 ms device delay.
      const [piece] = $voiceTimeline.get().segments[0].pieces
      expect([piece.startMs - now, piece.endMs - now].map(Math.round)).toEqual([250, 450])
      expect(getVoiceOutputLatencyMs()).toBe(200)
      expect(readVoiceLevel()).toBeCloseTo(0.4)
      expect($voice.get()).toEqual({ endReason: null, messageId: null, source: 'voice-conversation', status: 'speaking' })
    })

    it('gives a voice without word times one sentence that lasts as long as its audio', async () => {
      const ws = await streaming()

      ws.serverFrame({ type: 'segment', text: 'One.', units: [] })
      ws.serverPcm(4_800) // 100 ms
      ws.serverPcm(4_800)

      const [segment] = $voiceTimeline.get().segments
      expect(Math.round(segment.endMs - segment.startMs)).toBe(200)
      expect(segment.pieces[0]).toMatchObject({ endMs: segment.endMs, kind: 'sentence' })
    })

    it('keeps speaking until the last audio is heard, not just sent to the device', async () => {
      const ws = await streaming()

      ws.serverPcm(9_600) // 200 ms, heard 50..250 ms after a 200 ms device delay
      ws.serverFrame({ type: 'end' })
      await vi.advanceTimersByTimeAsync(400)
      expect($voice.get().status).toBe('speaking')

      await vi.advanceTimersByTimeAsync(200)
      expect($voice.get()).toMatchObject({ endReason: 'finished', status: 'idle' })
    })

    it('gives a whole clip in an audio element one sentence for as long as it plays', async () => {
      $voiceTimeline.set({ segments: [] })
      gateway.setNextUrl(null) // no streaming endpoint → data-url fallback

      const playback = playSpeechText('hello world', { messageId: 'm1', source: 'read-aloud' })
      await vi.advanceTimersByTimeAsync(0)

      const audio = FakeAudio.instances.at(-1)!

      const fire = (type: string) =>
        audio.addEventListener.mock.calls.filter(([name]) => name === type).forEach(([, listener]) => listener())

      expect($voiceTimeline.get().segments).toEqual([]) // nothing is heard yet

      Object.assign(audio, { duration: 2 })
      fire('playing')

      const [segment] = $voiceTimeline.get().segments
      expect(segment.pieces).toEqual([expect.objectContaining({ kind: 'sentence', text: 'hello world' })])
      expect(Math.round(segment.endMs - segment.startMs)).toBe(2_000)

      stopVoicePlayback()
      await playback
    })

    it('does the same for each sentence synthesized straight from the provider', async () => {
      $voiceTimeline.set({ segments: [] })
      direct.config = { mode: 'direct', provider: 'openai' }
      vi.stubGlobal('URL', { ...URL, createObjectURL: () => 'blob:x', revokeObjectURL: () => undefined })

      const session = await liveSession()

      session.append('Hello there, how are you today?')
      session.finish()
      await vi.advanceTimersByTimeAsync(0)

      const audio = FakeAudio.instances.at(-1)!

      audio.addEventListener.mock.calls.filter(([name]) => name === 'playing').forEach(([, listener]) => listener())
      expect($voiceTimeline.get().segments[0].pieces[0]).toMatchObject({ kind: 'sentence' })

      stopVoicePlayback()
      direct.config = null
      vi.unstubAllGlobals()
    })

    it('a stop drops what was still to come; a reply that plays out says finished', async () => {
      const ws = await streaming()

      ws.serverFrame({ type: 'segment', text: 'Hi', units: [{ endMs: 200, kind: 'word', startMs: 0, text: 'Hi' }] })
      ws.serverPcm(64)
      stopVoicePlayback()
      expect($voiceTimeline.get().segments).toEqual([])
      expect($voice.get().endReason).toBe('stopped')
      expect(readVoiceLevel()).toBe(0)

      const next = await streaming()

      next.serverPcm(64)
      next.serverFrame({ type: 'end' })
      await vi.advanceTimersByTimeAsync(500)
      expect($voice.get().endReason).toBe('finished')

      stopVoicePlayback() // from silence: nothing ended
      expect($voice.get().endReason).toBe('finished')
    })
  })
})
