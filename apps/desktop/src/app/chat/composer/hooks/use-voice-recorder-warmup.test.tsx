import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { MicRecording } from './use-mic-recorder'
import { useVoiceRecorder } from './use-voice-recorder'

// #105955 review: the mic-open warm-up must act as a READINESS BARRIER before
// transcription — `stop()` awaits the warm-up fired at start, so a cold local
// model load settles inside the lease request's own budget instead of eating
// the transcription request's decode deadline. Without the barrier, a 190 s
// cold load behind a 3 s recording exhausted the 180 s transcription floor
// before decoding began.

// The real recorder flips `recording` in start/stop; dictate() routes on it.
let recording = false

const micHandle = {
  cancel: vi.fn(),
  start: vi.fn(async () => {
    recording = true
  }),
  stop: vi.fn<() => Promise<MicRecording | null>>(async () => {
    recording = false

    return { audio: new Blob(), durationMs: 500, heardSpeech: true }
  })
}

vi.mock('./use-mic-recorder', () => ({
  useMicRecorder: () => ({
    handle: micHandle,
    level: 0,
    get recording() {
      return recording
    }
  })
}))

const syncSttLeaseSpy = vi.fn(async (..._args: unknown[]): Promise<void> => undefined)

vi.mock('@/lib/stt-lease', () => ({
  syncSttLease: (...args: unknown[]) => syncSttLeaseSpy(...args),
  VOICE_INPUT_LEASE: 'desktop:voice-input:test'
}))

vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: vi.fn()
}))

vi.mock('@/store/desktop-metrics', () => ({
  recordFeatureUse: vi.fn()
}))

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      notifications: {
        voice: {
          noSpeechDetected: 'no speech',
          recordingFailed: 'recording failed',
          transcriptionFailed: 'transcription failed',
          transcriptionUnavailable: 'unavailable',
          tryRecordingAgain: 'try again',
          unavailable: 'unavailable'
        }
      }
    }
  })
}))

vi.mock('../scope', () => ({
  useComposerScope: () => ({ connectionId: undefined, profile: undefined })
}))

// The ambient selection resolveOwnerNow reads; tests flip it mid-recording.
const ambient = { connectionId: 'gateway-a' as null | string, profile: 'worker_alpha' as null | string }

vi.mock('@/hermes', () => ({
  resolveOwnerNow: (owner?: { connectionId?: string; profile?: string }) => ({
    connectionId: owner?.connectionId || ambient.connectionId,
    profile: owner?.profile || ambient.profile
  })
}))

const OWNER_A = { connectionId: 'gateway-a', profile: 'worker_alpha' }

describe('useVoiceRecorder STT readiness barrier', () => {
  beforeEach(() => {
    cleanup()
    recording = false
    ambient.connectionId = 'gateway-a'
    ambient.profile = 'worker_alpha'
    micHandle.start.mockClear()
    micHandle.stop.mockReset()
    micHandle.stop.mockImplementation(async () => {
      recording = false

      return { audio: new Blob(), durationMs: 500, heardSpeech: true }
    })
    syncSttLeaseSpy.mockReset()
    syncSttLeaseSpy.mockImplementation(async () => undefined)
  })

  afterEach(() => {
    cleanup()
  })

  function renderRecorder(
    onTranscribeAudio: (audio: Blob, owner?: unknown) => Promise<string>,
    onTranscript: (text: string) => void = () => undefined
  ) {
    return renderHook(() =>
      useVoiceRecorder({
        focusInput: () => undefined,
        maxRecordingSeconds: 30,
        onTranscript,
        onTranscribeAudio
      })
    )
  }

  it('does not submit the clip to transcription while the mic-open warm-up is still settling', async () => {
    // Model the cold load: the acquire stays on the wire after the mic opens.
    let settleWarmup!: () => void
    syncSttLeaseSpy.mockImplementation(
      () =>
        new Promise<void>(resolve => {
          settleWarmup = resolve
        })
    )

    const transcribe = vi.fn(async () => 'hello world')
    const hook = renderRecorder(transcribe)

    // Mic opens: recording starts without waiting on the warm-up.
    await act(async () => {
      hook.result.current.dictate()
    })
    await act(async () => {
      await Promise.resolve()
    })
    expect(micHandle.start).toHaveBeenCalledTimes(1)
    expect(syncSttLeaseSpy).toHaveBeenCalledWith('desktop:voice-input:test', true, OWNER_A)

    // User stops after a short clip — but the cold load is still in flight.
    let stopping: Promise<void> = Promise.resolve()
    await act(async () => {
      stopping = hook.result.current.dictate() ?? Promise.resolve()
      await Promise.resolve()
      await Promise.resolve()
    })

    // The barrier holds: the transcription request has not gone out yet,
    // so the load wait is not inside the transcribe request's budget.
    expect(transcribe).not.toHaveBeenCalled()

    settleWarmup()
    await act(async () => {
      await stopping
    })

    // Readiness established → the clip is submitted and transcribes.
    await waitFor(() => expect(transcribe).toHaveBeenCalledTimes(1))
    expect(await transcribe.mock.results[0].value).toBe('hello world')
  })

  it('transcribes without extra delay once the warm-up has already settled', async () => {
    syncSttLeaseSpy.mockImplementation(async () => undefined)

    const transcribe = vi.fn(async () => 'hello world')
    const hook = renderRecorder(transcribe)

    await act(async () => {
      hook.result.current.dictate()
    })
    await act(async () => {
      await Promise.resolve()
    })
    await act(async () => {
      hook.result.current.dictate() // stop
    })

    await waitFor(() => expect(transcribe).toHaveBeenCalledTimes(1))
  })

  // #128668 review: one owner per recording — resolved at mic-open, used for
  // the warm-up, the transcription and the release.
  it('keeps warm-up, transcription and release on the mic-open owner across a mid-recording switch', async () => {
    const transcribe = vi.fn(async (_audio: Blob, _owner?: unknown) => 'hello world')
    const hook = renderRecorder(transcribe)

    await act(async () => {
      hook.result.current.dictate()
    })
    await act(async () => {
      await Promise.resolve()
    })

    ambient.connectionId = null
    ambient.profile = null

    await act(async () => {
      hook.result.current.dictate() // stop
    })
    await waitFor(() => expect(transcribe).toHaveBeenCalledTimes(1))

    expect(transcribe.mock.calls[0][1]).toEqual(OWNER_A)
    expect(syncSttLeaseSpy.mock.calls).toEqual([
      ['desktop:voice-input:test', true, OWNER_A],
      ['desktop:voice-input:test', false, OWNER_A]
    ])
  })

  // #128668 review (P1): a warm-up settling after unmount must not transcribe
  // or insert text into a torn-down composer.
  it('drops the dictation when the composer unmounts while the warm-up is settling', async () => {
    let settleWarmup!: () => void
    syncSttLeaseSpy.mockImplementationOnce(
      () =>
        new Promise<void>(resolve => {
          settleWarmup = resolve
        })
    )

    const transcribe = vi.fn(async () => 'hello world')
    const onTranscript = vi.fn()
    const hook = renderRecorder(transcribe, onTranscript)

    await act(async () => {
      hook.result.current.dictate()
    })
    await act(async () => {
      await Promise.resolve()
    })

    let stopping: Promise<void> = Promise.resolve()
    await act(async () => {
      stopping = hook.result.current.dictate() ?? Promise.resolve()
      await Promise.resolve()
    })

    hook.unmount()
    // Unmount releases to the owner that acquired.
    expect(syncSttLeaseSpy).toHaveBeenLastCalledWith('desktop:voice-input:test', false, OWNER_A)

    settleWarmup()
    await stopping

    expect(transcribe).not.toHaveBeenCalled()
    expect(onTranscript).not.toHaveBeenCalled()
  })

  it('drops a transcript that settles after unmount', async () => {
    let finishTranscription!: (text: string) => void

    const transcribe = vi.fn(
      () =>
        new Promise<string>(resolve => {
          finishTranscription = resolve
        })
    )

    const onTranscript = vi.fn()
    const hook = renderRecorder(transcribe, onTranscript)

    await act(async () => {
      hook.result.current.dictate()
    })
    await act(async () => {
      await Promise.resolve()
    })

    let stopping: Promise<void> = Promise.resolve()
    await act(async () => {
      stopping = hook.result.current.dictate() ?? Promise.resolve()
    })
    await waitFor(() => expect(transcribe).toHaveBeenCalledTimes(1))

    hook.unmount()
    finishTranscription('late words')
    await stopping

    expect(onTranscript).not.toHaveBeenCalled()
  })
})
