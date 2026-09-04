import { act, renderHook } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const mocks = vi.hoisted(() => ({
  cancel: vi.fn(),
  focusInput: vi.fn(),
  notifyError: vi.fn(),
  onTranscript: vi.fn(),
  resolveStop: undefined as ((value: { audio: Blob; durationMs: number; heardSpeech: boolean } | null) => void) | undefined
}))

vi.mock('./use-mic-recorder', () => ({
  useMicRecorder: () => ({
    handle: {
      cancel: mocks.cancel,
      start: vi.fn().mockResolvedValue(true),
      stop: () => new Promise(resolve => { mocks.resolveStop = resolve })
    },
    level: 0,
    recording: true
  })
}))

vi.mock('@/i18n', () => ({
  useI18n: () => ({ t: { notifications: { voice: {
    transcriptionFailed: 'transcription failed',
    recordingFailed: 'recording failed'
  } } } })
}))

vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: mocks.notifyError
}))

import { useVoiceRecorder } from './use-voice-recorder'

describe('useVoiceRecorder transcription cancellation', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.resolveStop = undefined
  })

  it('does not focus or publish a late transcript after cancel', async () => {
    let rejectTranscription: ((error: Error) => void) | undefined

    const { result } = renderHook(() => useVoiceRecorder({
      focusInput: mocks.focusInput,
      maxRecordingSeconds: 10,
      onTranscript: mocks.onTranscript,
      onTranscribeAudio: () => new Promise((_resolve, reject) => { rejectTranscription = reject })
    }))

    let stopping: Promise<string | null>
    await act(async () => {
      stopping = result.current.stopRecording()
      result.current.cancelRecording()
      mocks.resolveStop?.({ audio: new Blob(['audio']), durationMs: 1, heardSpeech: true })
      await Promise.resolve()
      rejectTranscription?.(new Error('late failure'))
      await stopping
    })

    expect(mocks.notifyError).not.toHaveBeenCalled()
    expect(mocks.focusInput).not.toHaveBeenCalled()
    expect(mocks.onTranscript).not.toHaveBeenCalled()
    expect(result.current.voiceStatus).toBe('idle')
  })

  it('notifies and resets focus/status for a current transcription failure', async () => {
    const failure = new Error('current failure')
    let rejectTranscription: ((error: Error) => void) | undefined

    const { result } = renderHook(() => useVoiceRecorder({
      focusInput: mocks.focusInput,
      maxRecordingSeconds: 10,
      onTranscript: mocks.onTranscript,
      onTranscribeAudio: () => new Promise((_resolve, reject) => { rejectTranscription = reject })
    }))

    let stopping: Promise<string | null>
    await act(async () => {
      stopping = result.current.stopRecording()
      mocks.resolveStop?.({ audio: new Blob(['audio']), durationMs: 1, heardSpeech: true })
      await Promise.resolve()
      rejectTranscription?.(failure)
      await stopping
    })

    expect(mocks.notifyError).toHaveBeenCalledWith(failure, 'transcription failed')
    expect(mocks.focusInput).toHaveBeenCalledTimes(1)
    expect(result.current.voiceStatus).toBe('idle')
  })
})
