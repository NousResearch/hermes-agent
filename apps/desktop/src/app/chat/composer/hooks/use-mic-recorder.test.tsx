import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { type MicRecorderErrorCopy, useMicRecorder } from './use-mic-recorder'

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

class FakeAudioContext extends EventTarget {
  static instances: FakeAudioContext[] = []
  static throwOnConstruct = false
  /** 'suspended' = autoplay/renderer handed back a context that isn't running. */
  static initialState: AudioContextState = 'running'
  /** Chromium leaves resume() pending (not rejected) when it won't resume. */
  static resumeHangs = false
  state: AudioContextState = FakeAudioContext.initialState
  closing = deferred()

  constructor() {
    super()

    if (FakeAudioContext.throwOnConstruct) {
      throw new DOMException('too many contexts', 'NotSupportedError')
    }

    FakeAudioContext.instances.push(this)
  }

  /** Peak deviation from the 128 midline the fake mic reports (0 = silence, 42 = full scale). */
  static amplitude = 0

  createAnalyser() {
    return {
      fftSize: 0,
      getByteTimeDomainData: (data: Uint8Array) => {
        data.forEach((_, i) => {
          data[i] = 128 + (i % 2 === 0 ? FakeAudioContext.amplitude : -FakeAudioContext.amplitude)
        })
      }
    }
  }

  createMediaStreamSource() {
    return { connect: vi.fn() }
  }

  resume = vi.fn(async () => {
    if (FakeAudioContext.resumeHangs) {
      await new Promise(() => undefined)
    }

    this.state = 'running'
  })

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
  FakeAudioContext.initialState = 'running'
  FakeAudioContext.resumeHangs = false
  FakeAudioContext.amplitude = 0
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

  // Speech captured while the meter's context is suspended reads as flat
  // silence; the take must say so instead of claiming "no speech heard".
  it('marks the take unverified when the meter context never starts running', async () => {
    FakeAudioContext.initialState = 'suspended'
    FakeAudioContext.resumeHangs = true
    const onMeterFailure = vi.fn()
    const { result } = renderHook(() => useMicRecorder(copy))

    await act(async () => {
      await result.current.handle.start({ onMeterFailure, onSilence: vi.fn(), silenceLevel: 0.075, silenceMs: 1_250 })
    })

    let recording: Awaited<ReturnType<typeof result.current.handle.stop>> = null

    await act(async () => {
      recording = await result.current.handle.stop()
    })

    expect(FakeAudioContext.instances[0].resume).toHaveBeenCalled()
    expect(recording).toMatchObject({ heardSpeech: false, meterUnverified: true })
    // Not a dead device: the take isn't reported as a meter failure.
    expect(onMeterFailure).not.toHaveBeenCalled()
  })

  it('waits for a suspended meter to resume before the take counts as metered', async () => {
    FakeAudioContext.initialState = 'suspended'
    const { result } = renderHook(() => useMicRecorder(copy))

    await act(async () => {
      await result.current.handle.start()
    })

    expect(FakeAudioContext.instances[0].state).toBe('running')

    let recording: Awaited<ReturnType<typeof result.current.handle.stop>> = null

    await act(async () => {
      recording = await result.current.handle.stop()
    })

    expect(recording).toMatchObject({ meterUnverified: false })
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


// A USB mic pops (and a start chime rings) the instant capture opens: a few tens of
// ms over the speech threshold, then silence. Counting that single loud frame as
// speech started the 1.2s end-of-utterance clock before the user said a word, so the
// take ended as just the pop — STT returned "[clicking]" and the real sentence, begun
// a beat later, was lost. Speech must be sustained, and the open transient ignored.
describe('useMicRecorder speech onset', () => {
  let frames: FrameRequestCallback[] = []
  let now = 0

  beforeEach(() => {
    frames = []
    now = 1_000_000
    vi.spyOn(Date, 'now').mockImplementation(() => now)
    vi.stubGlobal(
      'requestAnimationFrame',
      vi.fn((callback: FrameRequestCallback) => {
        frames.push(callback)

        return frames.length
      })
    )
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  /** Hold `amplitude` for `ms`, stepping the meter one 16 ms frame at a time. */
  const hold = (amplitude: number, ms: number) => {
    FakeAudioContext.amplitude = amplitude

    for (let elapsed = 0; elapsed < ms; elapsed += 16) {
      now += 16
      const next = frames.shift()

      next?.(now)
    }
  }

  const LOUD = 20 // normalized ≈ 0.48, far over silenceLevel 0.075

  async function startTake(onSilence = vi.fn()) {
    const { result } = renderHook(() => useMicRecorder(copy))

    await act(async () => {
      await result.current.handle.start({ onSilence, silenceLevel: 0.075, silenceMs: 1_250 })
    })

    return { onSilence, result }
  }

  it('does not treat the pop at mic-open as the start of speech', async () => {
    const { onSilence, result } = await startTake()

    hold(LOUD, 80)
    hold(0, 2_000)

    expect(onSilence).not.toHaveBeenCalled()

    let recording: Awaited<ReturnType<typeof result.current.handle.stop>> = null

    await act(async () => {
      recording = await result.current.handle.stop()
    })

    expect(recording).toMatchObject({ heardSpeech: false })
  })

  it('does not treat a short click mid-take as speech', async () => {
    const { onSilence } = await startTake()

    hold(0, 800)
    hold(LOUD, 60)
    hold(0, 2_000)

    expect(onSilence).not.toHaveBeenCalled()
  })

  it('still ends the take after sustained speech goes quiet', async () => {
    const { onSilence, result } = await startTake()

    hold(0, 500)
    hold(LOUD, 400)
    hold(0, 1_400)

    expect(onSilence).toHaveBeenCalledTimes(1)

    let recording: Awaited<ReturnType<typeof result.current.handle.stop>> = null

    await act(async () => {
      recording = await result.current.handle.stop()
    })

    expect(recording).toMatchObject({ heardSpeech: true })
  })
})
