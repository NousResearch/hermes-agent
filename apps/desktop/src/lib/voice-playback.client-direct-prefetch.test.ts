// Prefetch contract for the client-direct TTS path.
//
// The session used to synthesize sentence N+1 only after sentence N fired
// `ended`, so every sentence boundary carried a full synthesis of silence
// (0.7-1.1 s against a local CPU TTS server). `startSpeechStream` now
// prefetches: the next sentence is synthesized while the current one plays,
// bounded by MAX_PREFETCH so a long reply cannot buffer itself in memory.

import type * as SharedTypes from '@hermes/shared'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { startSpeechStream, stopVoicePlayback } from './voice-playback'

const direct = vi.hoisted(() => {
  const config = {
    mode: 'direct' as const,
    wire: 'openai-speech' as const,
    provider: 'kokoro',
    base_url: 'http://tts.test/v1',
    api_key: 'local',
    model: 'kokoro',
    voice: 'af_heart',
    speed: 1.25,
    min_len: 1,
    extra_body: {}
  }

  return { config, synthesize: vi.fn(async () => new ArrayBuffer(8)) }
})

vi.mock('@/lib/voice-client-direct', () => ({
  directTtsConfig: vi.fn(async () => direct.config),
  synthesizeSpeechClientDirect: (...args: unknown[]) => direct.synthesize(...(args as []))
}))

vi.mock('@hermes/shared', async importOriginal => {
  const actual = await importOriginal<typeof SharedTypes>()

  return { ...actual, resolveGatewayWsUrl: async () => null }
})

vi.mock('@/hermes', () => ({
  getApiRequestConnection: () => null,
  getApiRequestProfile: () => null,
  speakText: vi.fn(async () => ({ data_url: 'data:audio/mpeg;base64,AAAA', mime_type: 'audio/mpeg', ok: true }))
}))

class FakeAudio {
  static instances: FakeAudio[] = []

  paused = false
  src: string
  listeners = new Map<string, Array<() => void>>()

  addEventListener = vi.fn((type: string, callback: () => void) => {
    const registered = this.listeners.get(type) ?? []

    registered.push(callback)
    this.listeners.set(type, registered)
  })
  removeEventListener = vi.fn()
  play = vi.fn(async () => {
    this.paused = false
  })
  pause = vi.fn(() => {
    this.paused = true
  })
  load = vi.fn()

  /** Fire `ended` once, like a finished audio element. */
  ended() {
    const registered = this.listeners.get('ended') ?? []

    this.listeners.set('ended', [])

    for (const callback of registered) {
      callback()
    }
  }

  constructor(src: string) {
    this.src = src
    FakeAudio.instances.push(this)
  }
}

const tick = async (turns = 4) => {
  for (let index = 0; index < turns; index += 1) {
    await new Promise(resolve => setTimeout(resolve, 0))
  }
}

async function openSession() {
  const session = await startSpeechStream({ source: 'voice-conversation' })

  if (!session) {
    throw new Error('expected a client-direct speech session')
  }

  return session
}

describe('client-direct TTS prefetch', () => {
  beforeEach(() => {
    FakeAudio.instances = []
    direct.synthesize.mockReset()
    direct.synthesize.mockImplementation(async () => new ArrayBuffer(8))
    vi.stubGlobal('Audio', FakeAudio)

    const RealURL = globalThis.URL

    class FakeURL extends RealURL {}

    Object.assign(FakeURL, {
      createObjectURL: vi.fn(() => 'blob:tts'),
      revokeObjectURL: vi.fn()
    })
    vi.stubGlobal('URL', FakeURL)
  })

  afterEach(() => {
    stopVoicePlayback()
    vi.unstubAllGlobals()
  })

  it('synthesizes the next sentence while the current one is still playing', async () => {
    const session = await openSession()

    // Trailing space so the last sentence cuts without waiting for `finish()`.
    session.append('First sentence here. Second sentence here. Third sentence here. ')

    await tick()

    // Sentence one is playing and the following ones were fetched without
    // waiting for its `ended` event — the whole point of the change.
    expect(FakeAudio.instances).toHaveLength(1)
    expect(direct.synthesize.mock.calls.length).toBeGreaterThan(1)

    FakeAudio.instances[0]?.ended()
    await tick()

    expect(FakeAudio.instances).toHaveLength(2)

    FakeAudio.instances[1]?.ended()
    await tick()

    expect(FakeAudio.instances).toHaveLength(3)

    FakeAudio.instances[2]?.ended()
    await tick()

    session.finish()

    await expect(session.done).resolves.toBe('done')
    expect(direct.synthesize.mock.calls.length).toBe(3)
  })

  it('keeps the prefetch buffer bounded instead of synthesizing the whole reply', async () => {
    const session = await openSession()

    session.append('One. Two. Three. Four. Five. Six. Seven. Eight. Nine. Ten.')

    await tick()

    // MAX_PREFETCH (2) buffered + the one playing = at most 3 in flight.
    expect(direct.synthesize.mock.calls.length).toBeLessThanOrEqual(3)

    stopVoicePlayback()
    await session.done
  })

  it('stops prefetching after a barge-in', async () => {
    const session = await openSession()

    session.append('First sentence here. Second sentence here. Third sentence here.')
    await tick()

    const callsBeforeBarge = direct.synthesize.mock.calls.length

    stopVoicePlayback()

    await expect(session.done).resolves.toBe('done')
    await tick()

    expect(direct.synthesize.mock.calls.length).toBe(callsBeforeBarge)
  })

  it('reports fallback when the provider rejects before any audio played', async () => {
    direct.synthesize.mockRejectedValueOnce(new Error('kokoro down'))

    const session = await openSession()

    // Trailing space so the sentence cuts immediately; the first synthesis rejects.
    session.append('Only sentence here. ')

    await expect(session.done).resolves.toBe('fallback')
    expect(FakeAudio.instances).toHaveLength(0)
  })
})
