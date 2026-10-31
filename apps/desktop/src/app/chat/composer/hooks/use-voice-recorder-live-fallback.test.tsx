import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { openDictationStream } from '@/lib/voice-stream'

import { useVoiceRecorder } from './use-voice-recorder'

// A live dictation session fails two ways — the host answers with an error, or
// nothing ever arrives and `voice-stream.ts`'s finalize deadline fails it
// (regression coverage for #147333). Both must cost neither the take nor the
// composer: the recorder falls back to transcribing the recorded blob, and the
// status leaves "transcribing" — the state whose every voice control is disabled
// and whose `dictate()` is a no-op.

let recording = false

const micHandle = {
  cancel: vi.fn(),
  start: vi.fn(async (options?: { onPcmRate?: (rate: number) => void }) => {
    recording = true
    // The meter reports the device rate once, before the first chunk.
    options?.onPcmRate?.(48_000)
  }),
  stop: vi.fn(async () => {
    recording = false

    return { audio: new Blob(['take']), durationMs: 900, heardSpeech: true }
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

const liveSession = {
  cancel: vi.fn(),
  pushAudio: vi.fn(),
  stop: vi.fn(async (): Promise<string> => {
    throw new Error('live transcription timed out after 60s (no final frame)')
  })
}

vi.mock('@/lib/voice-stream', () => ({
  openDictationStream: vi.fn(async () => liveSession)
}))

vi.mock('@/lib/voice-client-direct', () => ({
  fetchVoiceClientConfigFor: vi.fn(async () => ({ stt: { streaming: true }, tts: {} }))
}))

vi.mock('@/lib/stt-lease', () => ({
  syncSttLease: vi.fn(async () => undefined),
  VOICE_INPUT_LEASE: 'desktop:voice-input:test'
}))

const notifyError = vi.fn()

vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: (...args: unknown[]) => notifyError(...args)
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

vi.mock('@/hermes', () => ({
  resolveOwnerNow: () => ({ connectionId: 'gateway-a', profile: 'default' })
}))

describe('useVoiceRecorder live-session failure', () => {
  beforeEach(() => {
    cleanup()
    recording = false
    liveSession.stop.mockClear()
    notifyError.mockClear()
    micHandle.start.mockClear()
    micHandle.stop.mockClear()
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

  /** One take: open the mic (which opens the live stream), then stop it. */
  async function dictateThroughLiveSession(hook: ReturnType<typeof renderRecorder>) {
    // Re-read `dictate` per call: the first only starts the take, so the second
    // must see the render where the recorder is open.
    await act(async () => {
      hook.result.current.dictate()
    })
    await act(async () => {
      await Promise.resolve()
    })
    await act(async () => {
      hook.result.current.dictate()
    })
  }

  it('transcribes the recorded take and leaves transcribing when the live session fails', async () => {
    const transcribe = vi.fn(async (_audio: Blob) => 'hello world')
    const onTranscript = vi.fn()
    const hook = renderRecorder(transcribe, onTranscript)

    await dictateThroughLiveSession(hook)

    await waitFor(() => expect(onTranscript).toHaveBeenCalledWith('hello world'))

    // The live path ran and failed; the take survived it as the recorded blob.
    expect(vi.mocked(openDictationStream)).toHaveBeenCalledTimes(1)
    expect(liveSession.stop).toHaveBeenCalledTimes(1)
    expect(transcribe).toHaveBeenCalledTimes(1)
    expect(transcribe.mock.calls[0][0]).toBeInstanceOf(Blob)

    // And the composer is usable again: not left in "transcribing".
    await waitFor(() => expect(hook.result.current.voiceStatus).toBe('idle'))
    expect(notifyError).not.toHaveBeenCalled()
  })

  it('leaves transcribing even when the fallback transcription fails too', async () => {
    const transcribe = vi.fn(async () => {
      throw new Error('Transcription timed out')
    })
    const hook = renderRecorder(transcribe)

    await dictateThroughLiveSession(hook)

    await waitFor(() => expect(notifyError).toHaveBeenCalled())
    expect(hook.result.current.voiceStatus).toBe('idle')
  })
})
