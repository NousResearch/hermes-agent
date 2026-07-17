import { act, renderHook, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useVoiceRecorder } from './use-voice-recorder'

const mic = vi.hoisted(() => ({
  cancel: vi.fn(),
  start: vi.fn(),
  stop: vi.fn()
}))

const notifications = vi.hoisted(() => ({
  notify: vi.fn(),
  notifyError: vi.fn()
}))

vi.mock('./use-mic-recorder', () => ({
  useMicRecorder: () => ({
    handle: { cancel: mic.cancel, start: mic.start, stop: mic.stop },
    level: 0,
    recording: false
  })
}))

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      notifications: {
        voice: {
          noSpeechDetected: 'No speech detected',
          recordingFailed: 'Recording failed',
          transcriptionFailed: 'Transcription failed',
          transcriptionUnavailable: 'Transcription unavailable',
          tryRecordingAgain: 'Try again',
          unavailable: 'Unavailable'
        }
      }
    }
  })
}))

vi.mock('@/store/notifications', () => notifications)

beforeEach(() => {
  vi.clearAllMocks()
  mic.start.mockResolvedValue(undefined)
  mic.stop.mockResolvedValue({
    audio: new Blob(['voice'], { type: 'audio/webm' }),
    durationMs: 500,
    heardSpeech: true
  })
})

function renderRecorder(onTranscribeAudio = vi.fn().mockResolvedValue('transcribed words')) {
  const focusInput = vi.fn()
  const onIdle = vi.fn()
  const onTranscript = vi.fn()

  const hook = renderHook(() =>
    useVoiceRecorder({
      focusInput,
      maxRecordingSeconds: 60,
      onIdle,
      onTranscript,
      onTranscribeAudio
    })
  )

  return { ...hook, focusInput, onIdle, onTranscript, onTranscribeAudio }
}

describe('useVoiceRecorder dictation delivery', () => {
  it('starts on the first toggle and marks the stopped transcript for submission on the second', async () => {
    const { onTranscript, result } = renderRecorder()

    act(() => result.current.dictate())
    await waitFor(() => expect(mic.start).toHaveBeenCalledOnce())

    act(() => result.current.dictate(true))

    await waitFor(() => expect(onTranscript).toHaveBeenCalledWith('transcribed words', true))
    expect(mic.stop).toHaveBeenCalledOnce()
  })

  it('keeps the on-screen dictation toggle transcript-only', async () => {
    const { onTranscript, result } = renderRecorder()

    act(() => result.current.dictate())
    await waitFor(() => expect(mic.start).toHaveBeenCalledOnce())
    act(() => result.current.dictate())

    await waitFor(() => expect(onTranscript).toHaveBeenCalledWith('transcribed words', false))
  })

  it('never submits an empty transcription', async () => {
    const { onTranscript, result } = renderRecorder(vi.fn().mockResolvedValue('   '))

    act(() => result.current.dictate())
    await waitFor(() => expect(mic.start).toHaveBeenCalledOnce())
    act(() => result.current.dictate(true))

    await waitFor(() => expect(notifications.notify).toHaveBeenCalledOnce())
    expect(onTranscript).not.toHaveBeenCalled()
  })

  it('queues a fast second press while microphone startup is still pending', async () => {
    let resolveStart: (() => void) | undefined
    mic.start.mockImplementationOnce(
      () =>
        new Promise<void>(resolve => {
          resolveStart = resolve
        })
    )
    const { onTranscript, result } = renderRecorder()

    act(() => result.current.dictate())
    act(() => result.current.dictate(true))

    expect(mic.start).toHaveBeenCalledOnce()
    expect(mic.stop).not.toHaveBeenCalled()

    await act(async () => resolveStart?.())

    await waitFor(() => expect(onTranscript).toHaveBeenCalledWith('transcribed words', true))
    expect(mic.start).toHaveBeenCalledOnce()
    expect(mic.stop).toHaveBeenCalledOnce()
  })

  it('returns to idle and releases ownership after an asynchronous recorder error', async () => {
    const { onIdle, result } = renderRecorder()

    act(() => result.current.dictate())
    await waitFor(() => expect(mic.start).toHaveBeenCalledOnce())
    const startOptions = mic.start.mock.calls[0]?.[0] as { onError?: (error: Error) => void }

    act(() => startOptions.onError?.(new Error('device disconnected')))

    await waitFor(() => expect(result.current.voiceStatus).toBe('idle'))
    expect(onIdle).toHaveBeenCalledOnce()

    act(() => result.current.dictate())
    await waitFor(() => expect(mic.start).toHaveBeenCalledTimes(2))
  })
})
