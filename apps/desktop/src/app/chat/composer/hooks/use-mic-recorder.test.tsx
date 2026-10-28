import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { type MicRecorderErrorCopy, type MicRecording, useMicRecorder } from './use-mic-recorder'

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

// The level meter behind continuous voice mode is an AudioContext on the
// capture stream. #75329: a torn-down context was still closing when the next
// take opened another, the device errored, and the dead meter silently
// dropped every later utterance.

const copy: MicRecorderErrorCopy = {
  microphoneAccessDenied: 'denied',
  microphoneConstraintsUnsupported: 'constraints',
  microphoneInUse: 'in use',
  microphoneNeedsHttps: 'needs https',
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
  state: AudioContextState = 'running'
  currentTime = 40
  closing = deferred()

  constructor() {
    super()

    if (FakeAudioContext.throwOnConstruct) {
      throw new DOMException('too many contexts', 'NotSupportedError')
    }

    FakeAudioContext.instances.push(this)
  }

  createMediaStreamSource() {
    return { connect: vi.fn() }
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
  static instances: FakeMediaRecorder[] = []
  static deferStop = false
  static pendingStops: (() => void)[] = []
  mimeType = 'audio/webm'
  state: RecordingState = 'inactive'
  ondataavailable: ((event: { data: Blob }) => void) | null = null
  onstop: (() => void) | null = null
  onerror: ((event: Event) => void) | null = null

  constructor() {
    FakeMediaRecorder.instances.push(this)
  }

  emit(text: string) {
    this.ondataavailable?.({ data: new Blob([text], { type: this.mimeType }) })
  }

  start() {
    this.state = 'recording'
  }

  stop() {
    this.state = 'inactive'
    const deliver = () => {
      this.ondataavailable?.({ data: new Blob(['clip'], { type: 'audio/webm' }) })
      this.onstop?.()
    }

    if (FakeMediaRecorder.deferStop) {
      FakeMediaRecorder.pendingStops.push(deliver)
    } else {
      deliver()
    }
  }
}

const flush = () => act(async () => new Promise<void>(resolve => window.setTimeout(resolve, 0)))

beforeEach(() => {
  FakeAudioContext.instances = []
  FakeAudioContext.throwOnConstruct = false
  FakeMediaRecorder.deferStop = false
  FakeMediaRecorder.instances = []
  FakeMediaRecorder.pendingStops = []
  meter.onLevel = null
  meter.onError = null
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
  vi.useRealTimers()
  vi.restoreAllMocks()
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

    act(() => FakeAudioContext.instances[0].dispatchEvent(new Event('error')))
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

  it('ends an utterance using audio timestamps even when no animation frame runs', async () => {
    const onSilence = vi.fn()
    const { result } = renderHook(() => useMicRecorder(copy))

    await act(async () => {
      await result.current.handle.start({ onSilence, silenceLevel: 0.075, silenceMs: 1_250 })
    })
    act(() => {
      meter.onLevel?.(0.2, 40_017)
      meter.onLevel?.(0, 40_033)
      meter.onLevel?.(0, 41_282)
    })
    expect(onSilence).not.toHaveBeenCalled()
    act(() => meter.onLevel?.(0, 41_283))
    expect(onSilence).toHaveBeenCalledOnce()

    let recording: Awaited<ReturnType<typeof result.current.handle.stop>> = null

    await act(async () => {
      recording = await result.current.handle.stop()
    })
    expect(recording).toMatchObject({ heardSpeech: true, meterFailed: false })
  })

  it('measures idle silence from this context’s audio-clock origin', async () => {
    const onSilence = vi.fn()
    const { result } = renderHook(() => useMicRecorder(copy))

    await act(async () => {
      await result.current.handle.start({ onSilence, silenceLevel: 0.075, idleSilenceMs: 1_000 })
    })
    act(() => meter.onLevel?.(0, 40_999))
    expect(onSilence).not.toHaveBeenCalled()
    act(() => {
      FakeAudioContext.instances[0].currentTime = 41 + 128 / 48_000
      meter.onLevel?.(0, 41_000)
    })
    expect(onSilence).toHaveBeenCalledOnce()
  })

  it.each(['idle', 'utterance'])('drains historical %s silence before ending captured speech', async boundary => {
    const { result } = renderHook(() => useMicRecorder(copy))
    let stopped: Promise<MicRecording | null> | undefined
    const onSilence = vi.fn(() => {
      stopped = result.current.handle.stop()
    })

    await act(async () => {
      await result.current.handle.start({
        onSilence,
        silenceLevel: 0.075,
        silenceMs: 1_250,
        idleSilenceMs: 12_000
      })
    })
    const context = FakeAudioContext.instances[0]
    const recorder = FakeMediaRecorder.instances[0]

    act(() => {
      if (boundary === 'utterance') {
        meter.onLevel?.(0.2, 40_017)
      }

      // The recorder already contains speech while quiet reports are queued.
      context.currentTime = 52.8
      recorder.emit('onset')
      meter.onLevel?.(0, 40_033)
      meter.onLevel?.(0, 52_000)
    })
    expect(onSilence).not.toHaveBeenCalled()
    expect(result.current.recording).toBe(true)
    act(() => {
      meter.onLevel?.(0.2, 52_200)
      meter.onLevel?.(0.2, 52_800)
      recorder.emit('command')
    })
    expect(onSilence).not.toHaveBeenCalled()
    act(() => {
      context.currentTime = 54.25
      meter.onLevel?.(0, 53_000)
      meter.onLevel?.(0, 54_250)
    })

    expect(onSilence).toHaveBeenCalledOnce()
    const recording = await stopped
    expect(recording).toMatchObject({ heardSpeech: true, meterFailed: false })
    expect(recording?.audio.size).toBe('onsetcommandclip'.length)
  })

  it('reports worklet failure instead of silently waiting for speech', async () => {
    const onMeterFailure = vi.fn()
    const { result } = renderHook(() => useMicRecorder(copy))

    await act(async () => {
      await result.current.handle.start({ onMeterFailure })
    })
    act(() => meter.onError?.())
    await flush()
    expect(onMeterFailure).toHaveBeenCalledOnce()

    let recording: Awaited<ReturnType<typeof result.current.handle.stop>> = null

    await act(async () => {
      recording = await result.current.handle.stop()
    })
    expect(recording).toMatchObject({ meterFailed: true })
  })

  it('does not deliver a deferred failure from a cancelled take into the next take', async () => {
    vi.useFakeTimers()
    const oldFailure = vi.fn()
    const nextFailure = vi.fn()
    const { result } = renderHook(() => useMicRecorder(copy))

    await act(async () => {
      await result.current.handle.start({ onMeterFailure: oldFailure })
    })
    act(() => {
      meter.onError?.()
      result.current.handle.cancel()
      FakeAudioContext.instances[0].closing.resolve()
    })
    await act(async () => {
      await result.current.handle.start({ onMeterFailure: nextFailure })
      await vi.runOnlyPendingTimersAsync()
    })
    expect(oldFailure).not.toHaveBeenCalled()
    expect(nextFailure).not.toHaveBeenCalled()
    expect(result.current.recording).toBe(true)
  })

  it('does not let a cancelled stop event tear down a newer take', async () => {
    FakeMediaRecorder.deferStop = true
    const { result } = renderHook(() => useMicRecorder(copy))

    await act(async () => {
      await result.current.handle.start()
    })
    let stopped!: ReturnType<typeof result.current.handle.stop>

    act(() => {
      stopped = result.current.handle.stop()
      result.current.handle.cancel()
      FakeAudioContext.instances[0].closing.resolve()
    })
    await act(async () => {
      await result.current.handle.start()
    })
    act(() => FakeMediaRecorder.pendingStops.splice(0).forEach(deliver => deliver()))

    expect(await stopped).toBeNull()
    expect(result.current.recording).toBe(true)
    expect(FakeAudioContext.instances[1].state).toBe('running')
  })

  it('keeps functional samples while hidden and refreshes the visual meter when visible again', async () => {
    const visibility = vi.spyOn(document, 'visibilityState', 'get').mockReturnValue('hidden')
    const onSilence = vi.fn()
    let renders = 0
    const { result } = renderHook(() => {
      renders++

      return useMicRecorder(copy)
    })

    await act(async () => {
      await result.current.handle.start({ onSilence, silenceLevel: 0.075, silenceMs: 1_250 })
    })
    const before = renders
    act(() => meter.onLevel?.(0.2, 40_017))
    expect(renders).toBe(before)
    visibility.mockReturnValue('visible')
    act(() => meter.onLevel?.(0.3, 40_033))
    expect(result.current.level).toBe(0.3)
    visibility.mockReturnValue('hidden')
    act(() => {
      meter.onLevel?.(0, 40_050)
      meter.onLevel?.(0, 41_300)
    })
    expect(onSilence).toHaveBeenCalledOnce()
  })
})
