import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { setDocumentHidden } from '@/test/window-state'

import { type MicRecorderErrorCopy, useMicRecorder } from './use-mic-recorder'

// A minimized or occluded Chromium window never runs requestAnimationFrame
// callbacks and throttles timers to ~1 Hz, while the audio graph keeps
// processing. These tests hide the window, make rAF inert, and feed audio
// through the fake graph: end-of-speech must still be detected.

const copy: MicRecorderErrorCopy = {
  microphoneAccessDenied: 'access denied',
  microphoneConstraintsUnsupported: 'constraints unsupported',
  microphoneInUse: 'in use',
  microphonePermissionDenied: 'permission denied',
  microphoneStartFailed: 'start failed',
  microphoneUnsupported: 'unsupported',
  noMicrophone: 'no microphone'
}

const FRAME_MS = 40

let signal = 0
let processors: { onaudioprocess: ((event: AudioProcessingEvent) => void) | null; disconnect: () => void }[] = []

const node = () => ({ connect: vi.fn(), disconnect: vi.fn() })

class FakeAudioContext {
  state = 'running'
  sampleRate = 48_000
  destination = node()
  resume = vi.fn(async () => undefined)
  close = vi.fn(async () => undefined)

  createMediaStreamSource() {
    return node()
  }

  createGain() {
    return { ...node(), gain: { value: 1 } }
  }

  createAnalyser() {
    return {
      ...node(),
      fftSize: 2048,
      getByteTimeDomainData: (data: Uint8Array) => data.fill(Math.max(0, Math.min(255, Math.round(128 + 128 * signal))))
    }
  }

  createScriptProcessor() {
    const processor = { ...node(), onaudioprocess: null }
    processors.push(processor)

    return processor
  }
}

class FakeMediaRecorder {
  static isTypeSupported = () => true
  mimeType = 'audio/webm'
  state: RecordingState = 'inactive'
  ondataavailable: ((event: BlobEvent) => void) | null = null
  onerror: ((event: Event) => void) | null = null
  onstop: (() => void) | null = null

  start() {
    this.state = 'recording'
  }

  stop() {
    this.state = 'inactive'
    this.onstop?.()
  }
}

/** Advance wall-clock time in audio-callback-sized steps while the graph "plays" `amplitude`. */
function feed(amplitude: number, durationMs: number) {
  signal = amplitude
  const samples = new Float32Array(2048).fill(amplitude)
  const event = { inputBuffer: { getChannelData: () => samples } } as unknown as AudioProcessingEvent

  for (let elapsed = 0; elapsed < durationMs; elapsed += FRAME_MS) {
    vi.setSystemTime(Date.now() + FRAME_MS)
    act(() => processors.forEach(processor => processor.onaudioprocess?.(event)))
  }
}

const SPEECH = 0.5
const SILENCE = 0

beforeEach(() => {
  signal = 0
  processors = []
  vi.useFakeTimers({ toFake: ['Date'] })
  vi.setSystemTime(new Date('2026-09-14T12:00:00Z'))
  setDocumentHidden(true)
  vi.stubGlobal('requestAnimationFrame', vi.fn(() => 1))
  vi.stubGlobal('cancelAnimationFrame', vi.fn())
  vi.stubGlobal('AudioContext', FakeAudioContext)
  vi.stubGlobal('MediaRecorder', FakeMediaRecorder)
  Object.defineProperty(navigator, 'mediaDevices', {
    configurable: true,
    value: { getUserMedia: vi.fn(async () => ({ getTracks: () => [{ stop: vi.fn() }] })) }
  })
})

afterEach(() => {
  cleanup()
  setDocumentHidden(false)
  vi.unstubAllGlobals()
  vi.useRealTimers()
})

async function startRecorder(options: Parameters<ReturnType<typeof useMicRecorder>['handle']['start']>[0]) {
  const { result } = renderHook(() => useMicRecorder(copy))

  await act(async () => {
    await result.current.handle.start(options)
  })

  return result
}

describe('useMicRecorder while the window is hidden', () => {
  it('detects end of speech without requestAnimationFrame', async () => {
    const onSilence = vi.fn()
    await startRecorder({ onSilence, silenceLevel: 0.075, silenceMs: 1_250 })

    feed(SPEECH, 600)
    expect(onSilence).not.toHaveBeenCalled()

    feed(SILENCE, 1_000)
    expect(onSilence).not.toHaveBeenCalled()

    feed(SILENCE, 400)
    expect(onSilence).toHaveBeenCalledTimes(1)

    feed(SILENCE, 2_000)
    expect(onSilence).toHaveBeenCalledTimes(1)
  })

  it('fires the idle timeout when nothing is said', async () => {
    const onSilence = vi.fn()
    await startRecorder({ idleSilenceMs: 3_000, onSilence, silenceLevel: 0.075, silenceMs: 1_250 })

    feed(SILENCE, 2_800)
    expect(onSilence).not.toHaveBeenCalled()

    feed(SILENCE, 400)
    expect(onSilence).toHaveBeenCalledTimes(1)
  })

  it('keeps reporting levels on the same scale as the analyser meter', async () => {
    const onLevel = vi.fn()
    await startRecorder({ onLevel })

    feed(0.1, 200)

    // byte time-domain RMS / 42: a constant 0.1 signal sits ~12.8 codes off centre.
    expect(onLevel).toHaveBeenLastCalledWith(expect.closeTo((0.1 * 128) / 42, 1))
  })

  it('stops metering once the recording stops', async () => {
    const onLevel = vi.fn()
    const result = await startRecorder({ onLevel })

    feed(SPEECH, 200)

    await act(async () => {
      await result.current.handle.stop()
    })

    const calls = onLevel.mock.calls.length
    feed(SPEECH, 400)

    expect(onLevel).toHaveBeenCalledTimes(calls)
  })
})
