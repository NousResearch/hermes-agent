// @vitest-environment jsdom
import { act, renderHook } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { notify } from '@/store/notifications'

import { useVoiceRecorder } from './use-voice-recorder'

const recorder = {
  start: vi.fn(),
  stop: vi.fn()
}

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      notifications: {
        voice: {
          unavailable: 'Voice input unavailable',
          transcriptionUnavailable: 'Transcription unavailable',
          recordingFailed: 'Recording failed',
          noSpeechDetected: 'No speech detected',
          tryRecordingAgain: 'Try recording again',
          transcriptionFailed: 'Transcription failed'
        }
      }
    }
  })
}))

vi.mock('@/store/notifications', () => ({ notify: vi.fn(), notifyError: vi.fn() }))

vi.mock('./use-mic-recorder', () => ({
  useMicRecorder: () => ({ handle: recorder, level: 0, recording: false })
}))

describe('useVoiceRecorder', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.useRealTimers()
    recorder.start.mockResolvedValue(undefined)
    recorder.stop.mockResolvedValue(null)
  })

  it('does not start an accidental 120-second recording before config arrives', async () => {
    const { result } = renderHook(() =>
      useVoiceRecorder({ maxRecordingSeconds: undefined, focusInput: vi.fn(), onTranscript: vi.fn(), onTranscribeAudio: vi.fn() })
    )

    await act(async () => {
      result.current.dictate()
    })

    expect(recorder.start).not.toHaveBeenCalled()
    expect(notify).toHaveBeenCalledWith({
      kind: 'warning',
      title: 'Voice input unavailable',
      message: 'Transcription unavailable'
    })
  })

  it('uses the configured duration for the browser hard cap', async () => {
    vi.useFakeTimers()
    const { result } = renderHook(() =>
      useVoiceRecorder({ maxRecordingSeconds: 240, focusInput: vi.fn(), onTranscript: vi.fn(), onTranscribeAudio: vi.fn() })
    )

    await act(async () => {
      result.current.dictate()
      await Promise.resolve()
    })

    await act(async () => {
      await vi.advanceTimersByTimeAsync(239_999)
    })
    expect(recorder.stop).not.toHaveBeenCalled()

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1)
    })
    expect(recorder.stop).toHaveBeenCalledOnce()
  })
})
