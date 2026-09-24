import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { useVoiceRecorder } from './use-voice-recorder'

// Transcript-source preference: a streaming transcript on the recording result
// wins over the one-shot file upload; without one the hook transcribes the
// captured audio exactly as before. The activity state always carries whatever
// live partial text the recorder exposes.

const mocks = vi.hoisted(() => ({
  handle: { cancel: vi.fn(), start: vi.fn(), stop: vi.fn() },
  state: { partialTranscript: '', recording: false }
}))

vi.mock('./use-mic-recorder', () => ({
  useMicRecorder: () => ({
    handle: mocks.handle,
    level: 0.4,
    partialTranscript: mocks.state.partialTranscript,
    recording: mocks.state.recording
  })
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

vi.mock('@/store/notifications', () => ({ notify: vi.fn(), notifyError: vi.fn() }))

function renderRecorder(onTranscribeAudio: (audio: Blob) => Promise<string>) {
  const onTranscript = vi.fn()
  const focusInput = vi.fn()

  const hook = renderHook(() =>
    useVoiceRecorder({ focusInput, maxRecordingSeconds: 300, onTranscript, onTranscribeAudio })
  )

  return { hook, onTranscript }
}

async function startThenStop(hook: ReturnType<typeof renderRecorder>['hook']) {
  await act(async () => {
    await hook.result.current.dictate()
  })
  await act(async () => {
    await hook.result.current.dictate()
  })
}

beforeEach(() => {
  mocks.handle.start.mockReset().mockImplementation(async () => {
    mocks.state.recording = true
  })
  mocks.handle.stop.mockReset()
  mocks.handle.cancel.mockReset()
  mocks.state.partialTranscript = ''
  mocks.state.recording = false
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('useVoiceRecorder transcript source', () => {
  it('prefers the streaming transcript and does not re-transcribe the audio', async () => {
    const audio = new Blob(['audio'])
    mocks.handle.stop.mockResolvedValueOnce({
      audio,
      durationMs: 12,
      heardSpeech: true,
      transcript: Promise.resolve('streamed words')
    })
    const onTranscribeAudio = vi.fn(async () => 'file words')
    const { hook, onTranscript } = renderRecorder(onTranscribeAudio)

    await startThenStop(hook)

    expect(onTranscribeAudio).not.toHaveBeenCalled()
    expect(onTranscript).toHaveBeenCalledWith('streamed words')
  })

  it('falls back to the file-based transcription when there is no stream transcript', async () => {
    const audio = new Blob(['audio'])
    mocks.handle.stop.mockResolvedValueOnce({ audio, durationMs: 12, heardSpeech: true })
    const onTranscribeAudio = vi.fn(async () => 'file words')
    const { hook, onTranscript } = renderRecorder(onTranscribeAudio)

    await startThenStop(hook)

    expect(onTranscribeAudio).toHaveBeenCalledWith(audio)
    expect(onTranscript).toHaveBeenCalledWith('file words')
  })

  it('plumbs the live partial transcript into the activity state', () => {
    mocks.state.partialTranscript = 'live partial'
    const { hook } = renderRecorder(vi.fn(async () => ''))

    expect(hook.result.current.voiceActivityState.partialTranscript).toBe('live partial')
  })
})
