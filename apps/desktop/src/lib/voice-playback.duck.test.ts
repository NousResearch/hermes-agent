import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $voicePlayback } from '@/store/voice-playback'

import { duckVoicePlayback, playSpeechText, stopVoicePlayback } from './voice-playback'

// #129843 — barge-in ducking must mute the live HTMLAudio timeline without
// disturbing it. jsdom's HTMLAudioElement can't play, so stub the constructor
// and drive `ended` by hand to pin the wiring: a clip created while ducked
// starts muted, ducking flips a playing clip in place, and a stop resets the
// duck so the NEXT clip is audible from its first sample.
class StubAudio {
  muted = false
  src = ''
  private ended: (() => void) | null = null

  constructor(src: string) {
    this.src = src
    created.push(this)
  }

  addEventListener(type: string, listener: () => void) {
    if (type === 'ended') {
      this.ended = listener
    }
  }

  removeEventListener() {
    this.ended = null
  }

  play() {
    return Promise.resolve()
  }

  pause() {}

  load() {}

  finish() {
    this.ended?.()
  }
}

vi.mock('@/lib/voice-client-direct', () => ({
  directTtsConfig: vi.fn(async () => null),
  synthesizeSpeechClientDirect: vi.fn()
}))

const speakTextMock = vi.fn(async () => ({ data_url: 'data:audio/wav;base64,eXh4' }))

vi.mock('@/hermes', () => ({
  getApiRequestConnection: () => null,
  getApiRequestProfile: () => null,
  speakText: (...args: unknown[]) => speakTextMock(...(args as []))
}))

const created: StubAudio[] = []

/** Start a clip and wait for its NEW audio element to exist (still "playing"). */
async function startClip(): Promise<{ audio: StubAudio; playback: Promise<boolean> }> {
  const before = created.length
  const playback = playSpeechText('hello there', { messageId: 'm1', source: 'read-aloud' })

  const audio = await vi.waitFor(() => {
    expect(created.length).toBe(before + 1)

    return created.at(-1)!
  })

  return { audio, playback }
}

describe('voice playback ducking (#129843)', () => {
  beforeEach(() => {
    created.length = 0
    vi.stubGlobal('Audio', StubAudio)
  })

  afterEach(async () => {
    stopVoicePlayback()
    $voicePlayback.set({
      audioElement: null,
      messageId: null,
      sequence: 0,
      source: null,
      status: 'idle'
    })
    vi.unstubAllGlobals()
    vi.clearAllMocks()
  })

  it('mutes and unmutes a playing clip in place without stopping it', async () => {
    const { audio, playback } = await startClip()

    duckVoicePlayback(true)
    expect(audio.muted).toBe(true)
    expect($voicePlayback.get().status).toBe('speaking')

    duckVoicePlayback(false)
    expect(audio.muted).toBe(false)
    expect($voicePlayback.get().status).toBe('speaking')

    audio.finish()
    await playback
  })

  it('resets the duck on stop so the next clip is audible from its first sample', async () => {
    const first = await startClip()

    duckVoicePlayback(true)
    expect(first.audio.muted).toBe(true)

    // stopVoicePlayback() ends the duck along with the clip (a new playback
    // must never inherit the muted state of the turn it replaced).
    stopVoicePlayback()
    await first.playback

    const second = await startClip()
    expect(second.audio.muted).toBe(false)

    second.audio.finish()
    await second.playback
  })
})
