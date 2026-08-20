import { act, renderHook, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { MicRecording } from './use-mic-recorder'
import { useVoiceRecorder } from './use-voice-recorder'

// Cancelling dictation must DISCARD the captured audio: the mic button only
// ever commits (stop → transcribe), so before this seam existed every started
// recording cost an STT call the user could not back out of.

const micHandle = {
  cancel: vi.fn(),
  start: vi.fn(async () => undefined),
  stop: vi.fn<() => Promise<MicRecording | null>>(async () => ({
    audio: new Blob(['audio']),
    durationMs: 1200,
    heardSpeech: true
  }))
}

let recording = false

vi.mock('./use-mic-recorder', () => ({
  useMicRecorder: () => ({ handle: micHandle, level: 0, recording })
}))

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      notifications: {
        voice: {
          noSpeechDetected: 'no speech',
          recordingFailed: 'recording failed',
          transcriptionFailed: 'transcription failed',
          transcriptionUnavailable: 'transcription unavailable',
          tryRecordingAgain: 'try again',
          unavailable: 'unavailable'
        }
      }
    }
  })
}))

vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: vi.fn()
}))

function renderRecorder() {
  const onTranscribeAudio = vi.fn(async () => 'transcribed text')
  const onTranscript = vi.fn()
  const focusInput = vi.fn()

  const view = renderHook(() =>
    useVoiceRecorder({ focusInput, maxRecordingSeconds: 120, onTranscribeAudio, onTranscript })
  )

  return { focusInput, onTranscribeAudio, onTranscript, view }
}

beforeEach(() => {
  recording = false
  micHandle.cancel.mockClear()
  micHandle.start.mockClear()
  micHandle.stop.mockClear()
})

describe('useVoiceRecorder cancelDictation', () => {
  it('discards the recording without transcribing it', async () => {
    const { focusInput, onTranscribeAudio, onTranscript, view } = renderRecorder()

    await act(async () => {
      await view.result.current.dictate()
    })

    recording = true
    view.rerender()

    await waitFor(() => expect(view.result.current.voiceStatus).toBe('recording'))

    act(() => {
      view.result.current.cancelDictation()
    })

    // The audio is dropped at the recorder, never committed through stop().
    expect(micHandle.cancel).toHaveBeenCalledTimes(1)
    expect(micHandle.stop).not.toHaveBeenCalled()
    expect(onTranscribeAudio).not.toHaveBeenCalled()
    expect(onTranscript).not.toHaveBeenCalled()

    expect(view.result.current.voiceStatus).toBe('idle')
    expect(view.result.current.voiceActivityState.elapsedSeconds).toBe(0)
    expect(focusInput).toHaveBeenCalled()
  })

  it('is a no-op when no dictation is running', () => {
    const { onTranscribeAudio, view } = renderRecorder()

    act(() => {
      view.result.current.cancelDictation()
    })

    expect(micHandle.cancel).not.toHaveBeenCalled()
    expect(onTranscribeAudio).not.toHaveBeenCalled()
    expect(view.result.current.voiceStatus).toBe('idle')
  })

  it('still transcribes on a normal stop — cancel does not change the commit path', async () => {
    const { onTranscribeAudio, onTranscript, view } = renderRecorder()

    await act(async () => {
      await view.result.current.dictate()
    })

    recording = true
    view.rerender()

    await act(async () => {
      await view.result.current.dictate()
    })

    expect(micHandle.stop).toHaveBeenCalledTimes(1)
    await waitFor(() => expect(onTranscribeAudio).toHaveBeenCalledTimes(1))
    await waitFor(() => expect(onTranscript).toHaveBeenCalledWith('transcribed text'))
    expect(micHandle.cancel).not.toHaveBeenCalled()
  })
})
