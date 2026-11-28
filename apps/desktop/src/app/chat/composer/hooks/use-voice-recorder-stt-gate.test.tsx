import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { MicRecording } from './use-mic-recorder'
import { useVoiceRecorder } from './use-voice-recorder'

// A dictation that can never be transcribed must not cost the user a take.
// `stt.enabled: false` makes the backend answer the transcript request with
// "STT is disabled in config.yaml", so the mic must stay shut and say why —
// the check lives in the recorder, not only on the button, so the
// `composer.dictate` keybind and tile mirrors are covered too.

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

vi.mock('@/lib/stt-lease', () => ({
  syncSttLease: vi.fn(async () => undefined),
  VOICE_INPUT_LEASE: 'desktop:voice-input:test'
}))

const notify = vi.fn()

vi.mock('@/store/notifications', () => ({
  notify: (...args: unknown[]) => notify(...args),
  notifyError: vi.fn()
}))

vi.mock('@/store/desktop-metrics', () => ({
  recordFeatureUse: vi.fn()
}))

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      desktop: { sttDisabled: 'Speech-to-text is disabled in settings.' },
      notifications: {
        voice: {
          noSpeechDetected: 'no speech',
          recordingFailed: 'recording failed',
          transcriptionFailed: 'transcription failed',
          transcriptionUnavailable: 'unavailable',
          tryRecordingAgain: 'try again',
          unavailable: 'Voice unavailable'
        }
      }
    }
  })
}))

vi.mock('../scope', () => ({
  useComposerScope: () => ({ connectionId: undefined, profile: undefined })
}))

vi.mock('@/hermes', () => ({
  resolveOwnerNow: () => ({ connectionId: 'gateway-a', profile: 'worker_alpha' })
}))

import { $sttEnabled } from '@/store/voice-prefs'

describe('useVoiceRecorder STT gate', () => {
  beforeEach(() => {
    cleanup()
    recording = false
    micHandle.start.mockClear()
    micHandle.stop.mockClear()
    notify.mockClear()
    $sttEnabled.set(true)
  })

  afterEach(() => {
    cleanup()
    $sttEnabled.set(true)
  })

  function renderRecorder() {
    return renderHook(() =>
      useVoiceRecorder({
        focusInput: () => undefined,
        maxRecordingSeconds: 30,
        onTranscript: () => undefined,
        onTranscribeAudio: async () => 'hello'
      })
    )
  }

  it('never opens the mic when stt.enabled is off, and says why', async () => {
    $sttEnabled.set(false)
    const hook = renderRecorder()

    await act(async () => {
      hook.result.current.dictate()
      await Promise.resolve()
    })

    expect(micHandle.start).not.toHaveBeenCalled()
    expect(hook.result.current.voiceStatus).toBe('idle')
    expect(notify).toHaveBeenCalledWith({
      kind: 'warning',
      title: 'Voice unavailable',
      message: 'Speech-to-text is disabled in settings.'
    })
  })

  it('opens the mic once stt.enabled is on again', async () => {
    $sttEnabled.set(true)
    const hook = renderRecorder()

    await act(async () => {
      hook.result.current.dictate()
      await Promise.resolve()
    })

    expect(micHandle.start).toHaveBeenCalledTimes(1)
    expect(notify).not.toHaveBeenCalled()
  })
})
