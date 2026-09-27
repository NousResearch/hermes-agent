import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { notifyError } from '@/store/notifications'

import type { MicRecorderOptions, MicRecording } from './use-mic-recorder'
import { useVoiceConversation } from './use-voice-conversation'

// #75329: continuous voice decides "did the user speak?" from the WebAudio
// level meter alone. When that meter's AudioContext hits a device error the
// take reads heardSpeech=false, and every later utterance was dropped before
// STT with no error shown.

const mocks = vi.hoisted(() => ({
  handle: {
    cancel: vi.fn(),
    start: vi.fn(async (_options?: MicRecorderOptions) => undefined),
    stop: vi.fn<() => Promise<MicRecording | null>>(async () => null)
  }
}))

vi.mock('./use-mic-recorder', () => ({ useMicRecorder: () => ({ handle: mocks.handle, level: 0 }) }))
vi.mock('@/lib/voice-barge-in', () => ({ monitorSpeechDuringPlayback: () => vi.fn() }))
vi.mock('@/lib/thinking-sound', () => ({ startThinkingSound: vi.fn(), stopThinkingSound: vi.fn() }))
vi.mock('@/lib/voice-playback', () => ({
  markVoicePlaybackInterrupted: vi.fn(),
  playSpeechText: vi.fn(async () => true),
  startSpeechStream: vi.fn(async () => null),
  stopVoicePlayback: vi.fn()
}))
vi.mock('@/store/notifications', () => ({ notify: vi.fn(), notifyError: vi.fn() }))
vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      notifications: {
        voice: {
          couldNotStartSession: 'could not start',
          microphoneFailed: 'mic failed',
          recordingFailed: 'recording failed',
          transcriptionFailed: 'transcription failed'
        }
      }
    }
  })
}))

function deadMeterTake(durationMs: number): MicRecording {
  return { audio: new Blob(['voice'], { type: 'audio/webm' }), durationMs, heardSpeech: false, meterFailed: true }
}

/** The options the hook handed to the most recent mic start. */
function lastStartOptions(): MicRecorderOptions {
  return mocks.handle.start.mock.calls.at(-1)?.[0] ?? {}
}

function renderConversation() {
  const onFatalError = vi.fn()
  const onSubmit = vi.fn()
  const onTranscribeAudio = vi.fn(async () => 'what time is it')

  const hook = renderHook(() =>
    useVoiceConversation({
      busy: false,
      consumePendingResponse: vi.fn(),
      enabled: true,
      onFatalError,
      onSubmit,
      onTranscribeAudio,
      pendingResponse: () => null
    })
  )

  return { hook, onFatalError, onSubmit, onTranscribeAudio }
}

describe('useVoiceConversation with a failed level meter', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  afterEach(cleanup)

  it('sends the take to STT instead of dropping it when the meter dies mid-utterance', async () => {
    const { hook, onSubmit, onTranscribeAudio } = renderConversation()

    await act(async () => {
      await hook.result.current.start()
    })
    await waitFor(() => expect(hook.result.current.status).toBe('listening'))

    const take = deadMeterTake(2_400)
    mocks.handle.stop.mockResolvedValueOnce(take)

    await act(async () => {
      lastStartOptions().onMeterFailure?.()
    })

    await waitFor(() => expect(onTranscribeAudio).toHaveBeenCalledWith(take.audio))
    await waitFor(() => expect(onSubmit).toHaveBeenCalledWith('what time is it'))
  })

  it('re-arms on a too-short dead-meter take, then stops with an error when the meter dies again', async () => {
    const { hook, onFatalError, onTranscribeAudio } = renderConversation()

    await act(async () => {
      await hook.result.current.start()
    })
    await waitFor(() => expect(hook.result.current.status).toBe('listening'))

    mocks.handle.stop.mockResolvedValueOnce(deadMeterTake(120))

    await act(async () => {
      lastStartOptions().onMeterFailure?.()
    })

    // Too short to hold speech: not sent to STT, and the loop listens again.
    await waitFor(() => expect(mocks.handle.start).toHaveBeenCalledTimes(2))
    expect(onTranscribeAudio).not.toHaveBeenCalled()
    expect(onFatalError).not.toHaveBeenCalled()

    mocks.handle.stop.mockResolvedValueOnce(deadMeterTake(120))

    await act(async () => {
      lastStartOptions().onMeterFailure?.()
    })

    await waitFor(() => expect(onFatalError).toHaveBeenCalledOnce())
    expect(notifyError).toHaveBeenCalledWith(expect.objectContaining({ message: 'recording failed' }), 'mic failed')
    expect(mocks.handle.start).toHaveBeenCalledTimes(2)
  })
})
