import { act, renderHook, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const { recorderCancel, recorderStart, recorderStop } = vi.hoisted(() => ({
  recorderCancel: vi.fn(),
  recorderStart: vi.fn(),
  recorderStop: vi.fn()
}))

let onSilence: (() => void) | undefined

vi.mock('./use-mic-recorder', () => ({
  useMicRecorder: () => ({
    handle: {
      cancel: recorderCancel,
      start: vi.fn(async (options: { onSilence?: () => void }) => {
        onSilence = options.onSilence
        recorderStart()
      }),
      stop: recorderStop
    },
    level: 0
  })
}))

vi.mock('@/lib/voice-playback', () => ({
  playSpeechText: vi.fn(async () => undefined),
  stopVoicePlayback: vi.fn()
}))

vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: vi.fn()
}))

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      notifications: {
        voice: {
          configureSpeechToText: 'configure STT',
          couldNotStartSession: 'could not start',
          microphoneFailed: 'microphone failed',
          playbackFailed: 'playback failed',
          transcriptionFailed: 'transcription failed',
          unavailable: 'unavailable'
        }
      }
    }
  })
}))

const { useVoiceConversation } = await import('./use-voice-conversation')

describe('useVoiceConversation', () => {
  beforeEach(() => {
    onSilence = undefined
    recorderStart.mockClear()
    recorderStop.mockReset()
    recorderCancel.mockClear()
  })

  it('automatically resumes listening after the final spoken response', async () => {
    let response: { id: string; pending: boolean; text: string } | null = null
    const pendingResponse = () => response
    const consumePendingResponse = vi.fn()
    const onSubmit = vi.fn(async () => undefined)
    const onTranscribeAudio = vi.fn(async () => 'hello')

    recorderStop.mockResolvedValue({
      audio: new Blob(['voice'], { type: 'audio/webm' }),
      durationMs: 500,
      heardSpeech: true
    })

    const { rerender } = renderHook(
      ({ enabled }: { enabled: boolean }) =>
        useVoiceConversation({
          busy: false,
          consumePendingResponse,
          enabled,
          onSubmit,
          onTranscribeAudio,
          pendingResponse
        }),
      { initialProps: { enabled: false } }
    )

    rerender({ enabled: true })
    await waitFor(() => expect(recorderStart).toHaveBeenCalledTimes(1))

    response = { id: 'assistant-1', pending: false, text: 'Hello there.' }
    await act(async () => {
      onSilence?.()
    })

    await waitFor(() => expect(onSubmit).toHaveBeenCalledWith('hello'))
    await waitFor(() => expect(consumePendingResponse).toHaveBeenCalledTimes(2))
    await waitFor(() => expect(recorderStart).toHaveBeenCalledTimes(2))
  })
})