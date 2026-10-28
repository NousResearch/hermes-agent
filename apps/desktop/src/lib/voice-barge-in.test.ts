import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $voiceSilenceMs } from '@/store/voice-prefs'

import {
  bargeInTriggerLevels,
  DEFAULT_BARGE_IN_THRESHOLD_MULTIPLIER,
  monitorSpeechDuringPlayback
} from './voice-barge-in'

// Audio samples advance independently of Date.now() and animation frames.
const meter = vi.hoisted(() => ({
  onLevel: null as ((level: number, audioTimeMs: number) => void) | null,
  onError: null as (() => void) | null
}))

vi.mock('@/lib/voice-level-meter', () => ({
  startVoiceLevelMeter: (
    context: AudioContext,
    _source: MediaStreamAudioSourceNode,
    onLevel: (level: number, audioTimeMs: number, isCurrent: boolean) => void,
    onError: () => void
  ) => {
    meter.onLevel = (level, at) => onLevel(level, at, context.currentTime * 1_000 < at + 1_000 / 60)
    meter.onError = onError

    return () => {
      meter.onLevel = null
      meter.onError = null
    }
  }
}))

let now = 0
let captureClock = 0
let micLevel = 0

beforeEach(() => {
  now = 1_000_000
  captureClock = now
  micLevel = 0
  meter.onLevel = null
  meter.onError = null
  vi.spyOn(Date, 'now').mockReturnValue(1_000_000)
  vi.stubGlobal('requestAnimationFrame', () => 1)
  vi.stubGlobal('cancelAnimationFrame', () => undefined)
  vi.stubGlobal(
    'AudioContext',
    class {
      get currentTime() {
        return captureClock / 1_000
      }
      state = 'running'

      createMediaStreamSource() {
        return { connect: () => undefined }
      }

      close() {
        return Promise.resolve()
      }
    }
  )
  Object.defineProperty(navigator, 'mediaDevices', {
    configurable: true,
    value: { getUserMedia: async () => ({ getTracks: () => [{ stop: () => undefined }] }) }
  })
})

afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

async function flushMicrotasks() {
  for (let i = 0; i < 5; i += 1) {
    await Promise.resolve()
  }
}

function advance(ms: number) {
  for (let elapsed = 0; elapsed < ms; elapsed += 16) {
    now += 16
    captureClock = Math.max(captureClock, now)
    meter.onLevel?.(Math.round(micLevel * 42) / 42, now)
  }
}

/** Quiet room, then TTS starts and the user talks over it at `speechLevel`. */
async function talkOverPlayback(speechLevel: number, thresholdMultiplier?: number | null) {
  let playing = false
  const onSpeech = vi.fn()
  const stop = monitorSpeechDuringPlayback({ isPlaying: () => playing, onSpeech, thresholdMultiplier })

  await flushMicrotasks()
  advance(600) // calibrate the quiet floor
  playing = true
  advance(700) // past the playback-onset grace window
  micLevel = speechLevel
  advance(800)
  stop()

  return onSpeech.mock.calls.length > 0
}

describe('monitorSpeechDuringPlayback — voice.barge_in_threshold_multiplier (#117801)', () => {
  // ~0.095: a Bluetooth HFP headset's speech level during playback, below the
  // stock playback clamp.
  const HFP_SPEECH_LEVEL = 0.095

  it('a quiet headset cannot interrupt playback at the stock sensitivity', async () => {
    expect(await talkOverPlayback(HFP_SPEECH_LEVEL)).toBe(false)
    expect(await talkOverPlayback(HFP_SPEECH_LEVEL, DEFAULT_BARGE_IN_THRESHOLD_MULTIPLIER)).toBe(false)
  })

  it('a lower configured multiplier lets the same headset interrupt playback', async () => {
    expect(await talkOverPlayback(HFP_SPEECH_LEVEL, 1.5)).toBe(true)
  })

  it('a higher configured multiplier makes playback harder to interrupt', async () => {
    const loud = bargeInTriggerLevels().playbackMinTrigger + 0.03

    expect(await talkOverPlayback(loud)).toBe(true)
    expect(await talkOverPlayback(loud, DEFAULT_BARGE_IN_THRESHOLD_MULTIPLIER * 2)).toBe(false)
  })

  it('invalid multipliers keep the stock trigger levels', () => {
    for (const value of [null, undefined, 0, -1, Number.NaN, Number.POSITIVE_INFINITY]) {
      expect(bargeInTriggerLevels(value)).toEqual(bargeInTriggerLevels(DEFAULT_BARGE_IN_THRESHOLD_MULTIPLIER))
    }
  })
})

class CaptureRecorder {
  static instances: CaptureRecorder[] = []
  static isTypeSupported = () => true
  state: RecordingState = 'inactive'
  mimeType = 'audio/webm'
  ondataavailable: ((event: { data: Blob }) => void) | null = null
  onstop: (() => void) | null = null
  stops = 0

  constructor() {
    CaptureRecorder.instances.push(this)
  }

  start() {
    this.state = 'recording'
  }

  emit(text: string) {
    this.ondataavailable?.({ data: new Blob([text], { type: this.mimeType }) })
  }

  stop() {
    this.stops++
    this.state = 'inactive'
    queueMicrotask(() => {
      this.emit('tail')
      this.onstop?.()
    })
  }
}

function blobText(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(String(reader.result))
    reader.onerror = () => reject(reader.error)
    reader.readAsText(blob)
  })
}

describe('barge capture on the audio clock', () => {
  beforeEach(() => {
    CaptureRecorder.instances = []
    vi.stubGlobal('MediaRecorder', CaptureRecorder)
  })

  it('keeps the speech onset and delivers one complete utterance without paint frames', async () => {
    const onSpeech = vi.fn()
    const onUtterance = vi.fn()
    monitorSpeechDuringPlayback({ isPlaying: () => false, onSpeech, onUtterance })
    await flushMicrotasks()
    advance(600)
    const recorder = CaptureRecorder.instances[0]
    recorder.emit('onset')
    micLevel = 0.2
    advance(500)
    expect(onSpeech).toHaveBeenCalledOnce()
    micLevel = 0
    advance($voiceSilenceMs.get() + 500)
    await flushMicrotasks()

    expect(recorder.stops).toBe(1)
    expect(onUtterance).toHaveBeenCalledOnce()
    expect(await blobText(onUtterance.mock.calls[0][0])).toBe('onsettail')
    advance(2_000)
    expect(onUtterance).toHaveBeenCalledOnce()
  })

  it('rotates quiet pre-roll without losing the next speech onset', async () => {
    const onUtterance = vi.fn()
    monitorSpeechDuringPlayback({ isPlaying: () => false, onSpeech: vi.fn(), onUtterance })
    await flushMicrotasks()
    CaptureRecorder.instances[0].emit('old')
    advance(5_600)
    expect(CaptureRecorder.instances).toHaveLength(2)
    CaptureRecorder.instances[1].emit('onset')
    micLevel = 0.2
    advance(500)
    micLevel = 0
    advance($voiceSilenceMs.get() + 500)
    await flushMicrotasks()

    expect(onUtterance).toHaveBeenCalledOnce()
    expect(await blobText(onUtterance.mock.calls[0][0])).toBe('onsettail')
  })

  it('drains a quiet rotation boundary before deciding whether captured speech can be discarded', async () => {
    const onSpeech = vi.fn()
    const onUtterance = vi.fn()
    monitorSpeechDuringPlayback({ isPlaying: () => false, onSpeech, onUtterance })
    await flushMicrotasks()
    advance(4_800)
    const recorder = CaptureRecorder.instances[0]

    // Audio capture continues through a renderer stall: speech begins at 5.2s,
    // but the quiet 5s report is not delivered until capture has reached 5.8s.
    captureClock += 1_000
    recorder.emit('onset')
    advance(400)
    expect(CaptureRecorder.instances).toHaveLength(1)
    expect(recorder.stops).toBe(0)
    micLevel = 0.2
    advance(600)
    expect(onSpeech).toHaveBeenCalledOnce()
    expect(recorder.state).toBe('recording')
    recorder.emit('command')
    micLevel = 0
    advance($voiceSilenceMs.get() + 500)
    await flushMicrotasks()

    expect(onUtterance).toHaveBeenCalledOnce()
    expect(await blobText(onUtterance.mock.calls[0][0])).toBe('onsetcommandtail')
  })

  it('does not end a captured utterance at historical silence before queued speech resumes', async () => {
    const onUtterance = vi.fn()
    monitorSpeechDuringPlayback({ isPlaying: () => false, onSpeech: vi.fn(), onUtterance })
    await flushMicrotasks()
    advance(600)
    const recorder = CaptureRecorder.instances[0]
    recorder.emit('onset')
    micLevel = 0.2
    advance(500)
    const quietMs = $voiceSilenceMs.get() + 100
    captureClock += quietMs + 800
    micLevel = 0
    advance(quietMs)
    expect(recorder.stops).toBe(0)
    micLevel = 0.2
    advance(800)
    expect(recorder.state).toBe('recording')
    recorder.emit('continuation')
    micLevel = 0
    advance($voiceSilenceMs.get() + 500)
    await flushMicrotasks()

    expect(onUtterance).toHaveBeenCalledOnce()
    expect(await blobText(onUtterance.mock.calls[0][0])).toBe('onsetcontinuationtail')
  })

  it('bounds continuous speech by captured audio time, not the stalled wall clock', async () => {
    const onUtterance = vi.fn()
    monitorSpeechDuringPlayback({ isPlaying: () => false, onSpeech: vi.fn(), onUtterance })
    await flushMicrotasks()
    advance(600)
    micLevel = 0.2
    advance(29_500)
    expect(onUtterance).not.toHaveBeenCalled()
    advance(1_500)
    await flushMicrotasks()

    expect(CaptureRecorder.instances[0].stops).toBe(1)
    expect(onUtterance).toHaveBeenCalledOnce()
  })

  it.each([false, true])('waits for current audio at the duration cap, unless cancelled: %s', async cancelled => {
    const onUtterance = vi.fn()
    const stop = monitorSpeechDuringPlayback({ isPlaying: () => false, onSpeech: vi.fn(), onUtterance })
    await flushMicrotasks()
    advance(600)
    const recorder = CaptureRecorder.instances[0]
    recorder.emit('onset')
    micLevel = 0.2
    advance(29_500)

    captureClock += 2_000
    advance(1_000)
    expect(recorder.stops).toBe(0)
    expect(onUtterance).not.toHaveBeenCalled()

    if (cancelled) {
      stop()
    } else {
      advance(1_000)
    }

    await flushMicrotasks()
    expect(recorder.stops).toBe(1)
    expect(onUtterance).toHaveBeenCalledTimes(cancelled ? 0 : 1)
  })

  it('does not deliver an utterance when the owner cancels inside onSpeech', async () => {
    const onUtterance = vi.fn()
    const stop = monitorSpeechDuringPlayback({
      isPlaying: () => false,
      onSpeech: () => stop(),
      onUtterance
    })
    await flushMicrotasks()
    advance(600)
    micLevel = 0.2
    advance(500)
    await flushMicrotasks()

    expect(CaptureRecorder.instances[0].stops).toBe(1)
    expect(onUtterance).not.toHaveBeenCalled()
  })

  it('releases a pending conversation capture when the meter fails, without transcribing partial speech', async () => {
    const onUtterance = vi.fn()
    monitorSpeechDuringPlayback({ isPlaying: () => false, onSpeech: vi.fn(), onUtterance })
    await flushMicrotasks()
    advance(600)
    micLevel = 0.2
    advance(500)
    const fail = meter.onError!
    fail()
    fail()
    await flushMicrotasks()

    expect(onUtterance.mock.calls).toEqual([[null]])
    expect(CaptureRecorder.instances[0].stops).toBe(1)
  })

  it('cancels an endpoint whose native recorder stop event is still queued', async () => {
    const onUtterance = vi.fn()
    const stop = monitorSpeechDuringPlayback({ isPlaying: () => false, onSpeech: vi.fn(), onUtterance })
    await flushMicrotasks()
    advance(600)
    micLevel = 0.2
    advance(500)
    micLevel = 0
    advance($voiceSilenceMs.get() + 500)
    expect(CaptureRecorder.instances[0].state).toBe('inactive')
    stop()
    await flushMicrotasks()

    expect(onUtterance).not.toHaveBeenCalled()
  })
})
