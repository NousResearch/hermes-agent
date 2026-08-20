import { act, renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { useVoiceRecorder } from './use-voice-recorder'

// Real-seam counterpart to use-voice-recorder.test.tsx: no mock of
// useMicRecorder. The discard claim — "the captured audio never reaches the
// STT provider" — is a data-loss/cost boundary, so it must be proven against
// the real recorder lifecycle (MediaRecorder wiring, chunk buffer, onstop
// resolver), with only the browser primitives faked.

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      notifications: {
        voice: {
          microphoneAccessDenied: 'denied',
          microphoneConstraintsUnsupported: 'constraints',
          microphoneInUse: 'in use',
          microphonePermissionDenied: 'permission',
          microphoneStartFailed: 'start failed',
          microphoneUnsupported: 'unsupported',
          noMicrophone: 'no mic',
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

class FakeMediaRecorder {
  static instances: FakeMediaRecorder[] = []
  static isTypeSupported = () => true

  mimeType = 'audio/webm'
  ondataavailable: ((event: { data: Blob }) => void) | null = null
  onerror: ((event: Event) => void) | null = null
  onstop: (() => void) | null = null
  state: 'inactive' | 'recording' = 'inactive'

  constructor(_stream: MediaStream, options?: { mimeType?: string }) {
    if (options?.mimeType) {
      this.mimeType = options.mimeType
    }

    FakeMediaRecorder.instances.push(this)
  }

  start() {
    this.state = 'recording'
  }

  stop() {
    this.state = 'inactive'
    // Flush a captured chunk the way a real recorder does on stop, THEN fire
    // onstop. A cancelled recorder has these handlers detached, so the chunk
    // must go nowhere — that detachment is exactly what this file tests.
    this.ondataavailable?.({ data: new Blob(['captured-audio']) })
    this.onstop?.()
  }
}

const fakeTrack = { stop: vi.fn() }
const fakeStream = { getTracks: () => [fakeTrack] } as unknown as MediaStream

beforeEach(() => {
  FakeMediaRecorder.instances = []
  fakeTrack.stop.mockClear()

  vi.stubGlobal('MediaRecorder', FakeMediaRecorder)
  Object.defineProperty(navigator, 'mediaDevices', {
    configurable: true,
    value: { getUserMedia: vi.fn(async () => fakeStream) }
  })
  // No AudioContext in jsdom: startMeter degrades gracefully (level stays 0),
  // which is fine — the meter is not the seam under test.
})

afterEach(() => {
  vi.unstubAllGlobals()
})

function renderRecorder() {
  const onTranscribeAudio = vi.fn<(audio: Blob) => Promise<string>>(async () => 'transcribed text')
  const onTranscript = vi.fn()

  const view = renderHook(() =>
    useVoiceRecorder({ focusInput: vi.fn(), maxRecordingSeconds: 120, onTranscribeAudio, onTranscript })
  )

  return { onTranscribeAudio, onTranscript, view }
}

describe('useVoiceRecorder against the real recorder seam', () => {
  it('cancelDictation discards the live MediaRecorder capture without transcribing', async () => {
    const { onTranscribeAudio, onTranscript, view } = renderRecorder()

    await act(async () => {
      view.result.current.dictate()
      await waitFor(() => expect(FakeMediaRecorder.instances).toHaveLength(1))
    })

    await waitFor(() => expect(view.result.current.voiceStatus).toBe('recording'))
    const recorder = FakeMediaRecorder.instances[0]
    expect(recorder.state).toBe('recording')

    act(() => {
      view.result.current.cancelDictation()
    })

    // The real cancel path: recorder stopped, handlers detached first so the
    // flushed chunk is dropped, tracks released, and the audio never reaches
    // the transcription callback.
    expect(recorder.state).toBe('inactive')
    expect(fakeTrack.stop).toHaveBeenCalled()
    expect(onTranscribeAudio).not.toHaveBeenCalled()
    expect(onTranscript).not.toHaveBeenCalled()
    expect(view.result.current.voiceStatus).toBe('idle')
  })

  it('a normal stop through the same real seam still transcribes — cancel changed nothing it should not', async () => {
    const { onTranscribeAudio, onTranscript, view } = renderRecorder()

    await act(async () => {
      view.result.current.dictate()
      await waitFor(() => expect(FakeMediaRecorder.instances).toHaveLength(1))
    })

    await waitFor(() => expect(view.result.current.voiceStatus).toBe('recording'))

    await act(async () => {
      view.result.current.dictate()
    })

    await waitFor(() => expect(onTranscribeAudio).toHaveBeenCalledTimes(1))
    const audio = onTranscribeAudio.mock.calls[0][0] as Blob
    expect(await audio.text()).toBe('captured-audio')
    await waitFor(() => expect(onTranscript).toHaveBeenCalledWith('transcribed text'))
  })
})
