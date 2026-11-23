import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, type Mock, vi } from 'vitest'

import { useMicRecorder } from '@/app/chat/composer/hooks/use-mic-recorder'
import { en } from '@/i18n/en'

import { monitorSpeechDuringPlayback } from './voice-barge-in'

class CaptureContext extends EventTarget {
  static instances: CaptureContext[] = []
  static initiallySuspended = false
  state: AudioContextState = CaptureContext.initiallySuspended ? 'suspended' : 'running'
  currentTime = 0
  destination = {}
  loading = Promise.withResolvers<void>()
  audioWorklet = { addModule: () => this.loading.promise }
  source = { connect: vi.fn(), disconnect: vi.fn() }
  tap = {
    connect: vi.fn(),
    disconnect: vi.fn(),
    onaudioprocess: null as ((event: { inputBuffer: { getChannelData: () => Float32Array } }) => void) | null
  }

  constructor() {
    super()
    CaptureContext.instances.push(this)
  }

  createMediaStreamSource() {
    return this.source
  }

  createScriptProcessor() {
    return this.tap
  }

  changeState(state: AudioContextState) {
    this.state = state
    this.dispatchEvent(new Event('statechange'))
  }

  resume = vi.fn(async () => this.changeState('running'))
  close = async () => this.changeState('closed')
}

class CaptureNode {
  static instances: CaptureNode[] = []
  onprocessorerror: (() => void) | null = null
  port = {
    onmessage: null as ((event: { data: { level: number; audioTime: number; nextReportTime: number } }) => void) | null,
    close: vi.fn()
  }
  connect = vi.fn()
  disconnect = vi.fn()

  constructor(readonly context: CaptureContext) {
    CaptureNode.instances.push(this)
  }

  audio(level: number, durationMs: number) {
    for (let elapsed = 0; elapsed < durationMs; elapsed += 20) {
      this.context.currentTime += 0.02
      const audioTime = this.context.currentTime
      this.port.onmessage?.({ data: { level, audioTime, nextReportTime: audioTime + 1 / 60 } })
    }
  }
}

class CaptureRecorder {
  static instances: CaptureRecorder[] = []
  static isTypeSupported = () => true
  state: RecordingState = 'inactive'
  mimeType = 'audio/webm'
  ondataavailable: ((event: { data: Blob }) => void) | null = null
  onstop: (() => void) | null = null
  onerror: (() => void) | null = null

  constructor() {
    CaptureRecorder.instances.push(this)
  }

  start() {
    this.state = 'recording'
  }

  stop() {
    this.state = 'inactive'
    this.ondataavailable?.({ data: new Blob(['captured speech'], { type: this.mimeType }) })
    this.onstop?.()
  }
}

const tracks: { stop: Mock }[] = []

const flush = () =>
  act(async () => {
    for (let i = 0; i < 12; i += 1) {
      await Promise.resolve()
    }
  })

beforeEach(() => {
  CaptureContext.instances = []
  CaptureContext.initiallySuspended = false
  CaptureNode.instances = []
  CaptureRecorder.instances = []
  tracks.length = 0
  vi.stubGlobal('AudioContext', CaptureContext)
  vi.stubGlobal('AudioWorkletNode', CaptureNode)
  vi.stubGlobal('MediaRecorder', CaptureRecorder)
  Object.defineProperty(navigator, 'mediaDevices', {
    configurable: true,
    value: {
      getUserMedia: async () => {
        const track = { stop: vi.fn() }
        tracks.push(track)

        return { getTracks: () => [track] }
      }
    }
  })
})

afterEach(async () => {
  cleanup()
  await flush()
  vi.unstubAllGlobals()
})

it('does not announce or record a listening take until its detector can hear the first words', async () => {
  const { result } = renderHook(() => useMicRecorder(en.notifications.voice))
  let started = false
  let start!: Promise<void>
  act(() => {
    start = result.current.handle.start({ onSilence: vi.fn(), silenceLevel: 0.075 }).then(() => {
      started = true
    })
  })
  await flush()

  expect(started).toBe(false)
  expect(result.current.recording).toBe(false)
  expect(CaptureRecorder.instances[0].state).toBe('inactive')

  await act(async () => {
    CaptureContext.instances[0].loading.resolve()
    await start
  })
  expect(result.current.recording).toBe(true)
  act(() => CaptureNode.instances[0].audio(0.3, 1_100))
  await act(async () => {
    const take = await result.current.handle.stop()
    expect(take).toMatchObject({ heardSpeech: true, meterFailed: false })
    expect(take!.audio.size).toBeGreaterThan(0)
  })
})

it.each(['resolve', 'reject'] as const)('isolates replacement capture from a cancelled module %s', async outcome => {
  const { result } = renderHook(() => useMicRecorder(en.notifications.voice))
  const onMeterFailure = vi.fn()
  let first!: Promise<unknown>
  act(() => {
    first = result.current.handle.start({ onMeterFailure }).catch(error => error)
  })
  await flush()
  const oldContext = CaptureContext.instances[0]
  act(() => result.current.handle.cancel())
  await act(async () => {
    expect(await first).toMatchObject({ name: 'AbortError' })
  })
  expect(tracks[0].stop).toHaveBeenCalledOnce()

  let replacement!: Promise<void>
  act(() => {
    replacement = result.current.handle.start({ onMeterFailure, onSilence: vi.fn(), silenceLevel: 0.075 })
  })
  await flush()
  await act(async () => {
    CaptureContext.instances[1].loading.resolve()
    await replacement

    if (outcome === 'resolve') {
      oldContext.loading.resolve()
    } else {
      oldContext.loading.reject(new Error('late asset error'))
    }
  })
  expect(CaptureNode.instances).toHaveLength(1)
  expect(CaptureRecorder.instances[0].state).toBe('inactive')
  expect(result.current.recording).toBe(true)
  expect(onMeterFailure).not.toHaveBeenCalled()
  act(() => CaptureNode.instances[0].audio(0.3, 400))
  await act(async () => {
    expect(await result.current.handle.stop()).toMatchObject({ heardSpeech: true, meterFailed: false })
  })
})

it('releases a suspended barge capture without submitting its truncated command', async () => {
  CaptureContext.initiallySuspended = true
  const onSpeech = vi.fn()
  const onUtterance = vi.fn()
  const stop = monitorSpeechDuringPlayback({ onSpeech, onUtterance, isPlaying: () => false })

  try {
    await flush()
    const context = CaptureContext.instances[0]
    context.loading.resolve()
    await flush()
    expect(context.resume).toHaveBeenCalledOnce()
    const node = CaptureNode.instances[0]
    node.audio(0, 600)
    node.audio(0.3, 400)
    expect(onSpeech).toHaveBeenCalledOnce()
    expect(onUtterance).not.toHaveBeenCalled()

    context.changeState('suspended')
    await flush()
    expect(onUtterance).toHaveBeenCalledExactlyOnceWith(null)
    expect(CaptureRecorder.instances[0].state).toBe('inactive')
    expect(tracks[0].stop).toHaveBeenCalledOnce()
    expect(node.port.close).toHaveBeenCalledOnce()
    expect(context.resume).toHaveBeenCalledOnce()
  } finally {
    stop()
  }
})

it('preserves live PCM after module failure in an initially suspended context', async () => {
  CaptureContext.initiallySuspended = true
  const pcm: ArrayBuffer[] = []
  const { result } = renderHook(() => useMicRecorder(en.notifications.voice))
  let start!: Promise<void>
  act(() => {
    start = result.current.handle.start({ onPcm: chunk => pcm.push(chunk) })
  })
  await flush()
  const context = CaptureContext.instances[0]
  await act(async () => {
    context.loading.reject(new Error('worklet unavailable'))
    await start
  })
  expect(context.state).toBe('running')
  expect(result.current.recording).toBe(true)
  context.tap.onaudioprocess?.({ inputBuffer: { getChannelData: () => new Float32Array([0.5, -0.5]) } })
  expect(Array.from(new Int16Array(pcm[0]))).toEqual([16383, -16384])
  await act(async () => {
    expect(await result.current.handle.stop()).toMatchObject({ meterFailed: true })
  })
})
