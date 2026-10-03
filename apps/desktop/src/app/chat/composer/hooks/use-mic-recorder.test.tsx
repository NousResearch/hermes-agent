import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { setDocumentHidden } from '@/test/window-state'

import { type MicRecorderErrorCopy, type MicRecorderOptions, useMicRecorder } from './use-mic-recorder'

// The level meter behind continuous voice mode is an AudioContext on the
// capture stream. #75329: a torn-down context was still closing when the next
// take opened another, the device errored, and the dead meter silently
// dropped every later utterance.

const copy: MicRecorderErrorCopy = {
  microphoneAccessDenied: 'denied',
  microphoneConstraintsUnsupported: 'constraints',
  microphoneInUse: 'in use',
  microphonePermissionDenied: 'permission',
  microphoneStartFailed: 'start failed',
  microphoneUnsupported: 'unsupported',
  noMicrophone: 'no mic'
}

function deferred() {
  let resolve!: () => void
  const promise = new Promise<void>(done => (resolve = done))

  return { promise, resolve }
}

interface FakeProcessor {
  connect: ReturnType<typeof vi.fn>
  disconnect: ReturnType<typeof vi.fn>
  onaudioprocess: ((event: AudioProcessingEvent) => void) | null
}

let processors: FakeProcessor[] = []

class FakeAudioContext extends EventTarget {
  static instances: FakeAudioContext[] = []
  static throwOnConstruct = false
  state: AudioContextState = 'running'
  closing = deferred()
  destination = {}

  constructor() {
    super()

    if (FakeAudioContext.throwOnConstruct) {
      throw new DOMException('too many contexts', 'NotSupportedError')
    }

    FakeAudioContext.instances.push(this)
  }

  createAnalyser() {
    return { fftSize: 0, getByteTimeDomainData: (data: Uint8Array) => data.fill(128) }
  }

  createMediaStreamSource() {
    return { connect: vi.fn() }
  }

  createGain() {
    return { connect: vi.fn(), disconnect: vi.fn(), gain: { value: 1 } }
  }

  createScriptProcessor() {
    const processor: FakeProcessor = { connect: vi.fn(), disconnect: vi.fn(), onaudioprocess: null }

    processors.push(processor)

    return processor
  }

  resume = vi.fn(async () => undefined)

  close() {
    return this.closing.promise.then(() => {
      this.state = 'closed'
      this.dispatchEvent(new Event('statechange'))
    })
  }
}

class FakeMediaRecorder {
  static isTypeSupported = () => true
  mimeType = 'audio/webm'
  state: RecordingState = 'inactive'
  ondataavailable: ((event: { data: Blob }) => void) | null = null
  onstop: (() => void) | null = null
  onerror: ((event: Event) => void) | null = null

  start() {
    this.state = 'recording'
  }

  stop() {
    this.state = 'inactive'
    this.ondataavailable?.({ data: new Blob(['clip'], { type: 'audio/webm' }) })
    this.onstop?.()
  }
}

const flush = () => act(async () => new Promise<void>(resolve => window.setTimeout(resolve, 0)))

beforeEach(() => {
  FakeAudioContext.instances = []
  FakeAudioContext.throwOnConstruct = false
  processors = []
  vi.stubGlobal('AudioContext', FakeAudioContext)
  vi.stubGlobal('MediaRecorder', FakeMediaRecorder)
  vi.stubGlobal(
    'requestAnimationFrame',
    vi.fn(() => 1)
  )
  vi.stubGlobal('cancelAnimationFrame', vi.fn())
  Object.defineProperty(navigator, 'mediaDevices', {
    configurable: true,
    value: { getUserMedia: vi.fn(async () => ({ getTracks: () => [{ stop: vi.fn() }] })) }
  })
})

afterEach(() => {
  cleanup()
  FakeAudioContext.instances.forEach(context => context.closing.resolve())
  vi.unstubAllGlobals()
})

describe('useMicRecorder level meter', () => {
  it('waits for the previous take’s meter to finish closing before opening the next', async () => {
    const { result } = renderHook(() => useMicRecorder(copy))

    await act(async () => {
      await result.current.handle.start()
    })
    await act(async () => {
      await result.current.handle.stop()
    })

    const first = FakeAudioContext.instances[0]
    let secondStart: Promise<void> | undefined

    act(() => {
      secondStart = result.current.handle.start()
    })
    await flush()

    // Still closing: no second context on top of it.
    expect(FakeAudioContext.instances).toHaveLength(1)

    await act(async () => {
      first.closing.resolve()
      await secondStart
    })

    expect(FakeAudioContext.instances).toHaveLength(2)
    expect(first.state).toBe('closed')
  })

  it('reports a device error on the meter and marks the take meterFailed', async () => {
    const onMeterFailure = vi.fn()
    const { result } = renderHook(() => useMicRecorder(copy))

    await act(async () => {
      await result.current.handle.start({ onMeterFailure, onSilence: vi.fn(), silenceLevel: 0.075, silenceMs: 1_250 })
    })

    FakeAudioContext.instances[0].dispatchEvent(new Event('error'))
    await flush()

    expect(onMeterFailure).toHaveBeenCalledOnce()

    let recording: Awaited<ReturnType<typeof result.current.handle.stop>> = null

    await act(async () => {
      recording = await result.current.handle.stop()
    })

    expect(recording).toMatchObject({ heardSpeech: false, meterFailed: true })
  })

  it('treats a meter that cannot be built as failed instead of silently deaf', async () => {
    FakeAudioContext.throwOnConstruct = true
    const onMeterFailure = vi.fn()
    const { result } = renderHook(() => useMicRecorder(copy))

    await act(async () => {
      await result.current.handle.start({ onMeterFailure })
    })
    await flush()

    expect(onMeterFailure).toHaveBeenCalledOnce()
  })

  it('does not report its own close at the end of a take as a failure', async () => {
    const onMeterFailure = vi.fn()
    const { result } = renderHook(() => useMicRecorder(copy))

    await act(async () => {
      await result.current.handle.start({ onMeterFailure })
    })

    let recording: Awaited<ReturnType<typeof result.current.handle.stop>> = null

    await act(async () => {
      recording = await result.current.handle.stop()
      FakeAudioContext.instances[0].closing.resolve()
    })
    await flush()

    expect(onMeterFailure).not.toHaveBeenCalled()
    expect(recording).toMatchObject({ meterFailed: false })
  })
})

// A minimized or occluded Chromium window never runs requestAnimationFrame
// callbacks and throttles timers to ~1 Hz, while the audio graph keeps
// processing. These tests hide the window, keep rAF inert and feed audio
// through the fake graph: end of speech must still be detected.
describe('useMicRecorder while the window is hidden', () => {
  const FRAME_MS = 40
  const SPEECH = 0.5
  const SILENCE = 0

  /** Advance wall-clock time in audio-callback-sized steps while the graph "plays" `amplitude`. */
  function feed(amplitude: number, durationMs: number) {
    const samples = new Float32Array(2048).fill(amplitude)
    const event = { inputBuffer: { getChannelData: () => samples } } as unknown as AudioProcessingEvent

    for (let elapsed = 0; elapsed < durationMs; elapsed += FRAME_MS) {
      vi.setSystemTime(Date.now() + FRAME_MS)
      act(() => processors.forEach(processor => processor.onaudioprocess?.(event)))
    }
  }

  async function startRecorder(options: MicRecorderOptions) {
    const { result } = renderHook(() => useMicRecorder(copy))

    await act(async () => {
      await result.current.handle.start(options)
    })

    return result
  }

  beforeEach(() => {
    vi.useFakeTimers({ toFake: ['Date'] })
    vi.setSystemTime(new Date('2026-10-02T12:00:00Z'))
    setDocumentHidden(true)
  })

  afterEach(() => {
    setDocumentHidden(false)
    vi.useRealTimers()
  })

  it('detects end of speech without requestAnimationFrame', async () => {
    const onSilence = vi.fn()
    await startRecorder({ onSilence, silenceLevel: 0.075, silenceMs: 1_250 })

    feed(SPEECH, 600)
    expect(onSilence).not.toHaveBeenCalled()

    feed(SILENCE, 1_000)
    expect(onSilence).not.toHaveBeenCalled()

    feed(SILENCE, 400)
    expect(onSilence).toHaveBeenCalledOnce()

    feed(SILENCE, 2_000)
    expect(onSilence).toHaveBeenCalledOnce()
  })

  it('fires the idle timeout when nothing is said', async () => {
    const onSilence = vi.fn()
    await startRecorder({ idleSilenceMs: 3_000, onSilence, silenceLevel: 0.075, silenceMs: 1_250 })

    feed(SILENCE, 2_800)
    expect(onSilence).not.toHaveBeenCalled()

    feed(SILENCE, 400)
    expect(onSilence).toHaveBeenCalledOnce()
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
    expect(onLevel).toHaveBeenLastCalledWith(1)

    await act(async () => {
      await result.current.handle.stop()
    })

    const calls = onLevel.mock.calls.length
    feed(SPEECH, 400)

    expect(onLevel).toHaveBeenCalledTimes(calls)
  })

  it('stops metering when the meter fails', async () => {
    const onLevel = vi.fn()
    const onMeterFailure = vi.fn()
    await startRecorder({ onLevel, onMeterFailure })

    feed(SPEECH, 200)
    expect(onLevel).toHaveBeenLastCalledWith(1)

    FakeAudioContext.instances[0].dispatchEvent(new Event('error'))
    await flush()

    expect(onMeterFailure).toHaveBeenCalledOnce()

    const calls = onLevel.mock.calls.length
    feed(SPEECH, 400)

    expect(onLevel).toHaveBeenCalledTimes(calls)
  })
})
