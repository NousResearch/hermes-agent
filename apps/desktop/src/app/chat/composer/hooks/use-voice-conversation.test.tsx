import { act, renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { notify } from '@/store/notifications'

import { useVoiceConversation } from './use-voice-conversation'

const recorder = vi.hoisted(() => ({
  handle: {
    cancel: vi.fn(),
    start: vi.fn(),
    stop: vi.fn()
  },
  level: 0.42,
  recording: false
}))

const playback = vi.hoisted(() => ({
  playSpeechText: vi.fn(),
  stopVoicePlayback: vi.fn()
}))

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      notifications: {
        voice: {
          configureSpeechToText: 'Configure speech-to-text to use voice mode.',
          couldNotStartSession: 'Could not start voice session.',
          microphoneAccessDenied: 'Microphone access was denied.',
          microphoneConstraintsUnsupported: 'Microphone constraints are unsupported.',
          microphoneFailed: 'Microphone failed.',
          microphoneInUse: 'Microphone is already in use.',
          microphonePermissionDenied: 'Microphone permission was denied.',
          microphoneStartFailed: 'Microphone could not start.',
          microphoneUnsupported: 'Microphone recording is not supported.',
          noMicrophone: 'No microphone was found.',
          playbackFailed: 'Playback failed.',
          transcriptionFailed: 'Transcription failed.',
          unavailable: 'Voice unavailable'
        }
      }
    }
  })
}))

vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: vi.fn()
}))

vi.mock('@/lib/voice-playback', () => playback)

vi.mock('./use-mic-recorder', () => ({
  useMicRecorder: () => recorder
}))

describe('useVoiceConversation', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    recorder.handle.start.mockResolvedValue(undefined)
    recorder.handle.stop.mockResolvedValue({
      audio: new Blob(['voice'], { type: 'audio/webm' }),
      durationMs: 1200,
      heardSpeech: true
    })
    playback.playSpeechText.mockResolvedValue(true)
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('warns and exits gracefully when speech-to-text is unavailable', async () => {
    const onFatalError = vi.fn()

    const { result } = renderHook(() =>
      useVoiceConversation({
        busy: false,
        consumePendingResponse: vi.fn(),
        enabled: false,
        onFatalError,
        onSubmit: vi.fn(),
        pendingResponse: () => null
      })
    )

    await act(async () => {
      await result.current.start()
    })

    expect(notify).toHaveBeenCalledWith({
      kind: 'warning',
      message: 'Configure speech-to-text to use voice mode.',
      title: 'Voice unavailable'
    })
    expect(onFatalError).toHaveBeenCalledTimes(1)
    expect(recorder.handle.start).not.toHaveBeenCalled()
    expect(result.current.status).toBe('idle')
  })

  it('listens, transcribes, and submits a speech turn through the backend callback', async () => {
    const onSubmit = vi.fn().mockResolvedValue(undefined)
    const onTranscribeAudio = vi.fn().mockResolvedValue('  hello hermes  ')

    const { result } = renderHook(() =>
      useVoiceConversation({
        busy: false,
        consumePendingResponse: vi.fn(),
        enabled: true,
        onSubmit,
        onTranscribeAudio,
        pendingResponse: () => null
      })
    )

    await act(async () => {
      await result.current.start()
    })

    expect(recorder.handle.start).toHaveBeenCalledWith(
      expect.objectContaining({
        idleSilenceMs: 12_000,
        silenceLevel: 0.075,
        silenceMs: 1_250
      })
    )
    expect(result.current.status).toBe('listening')

    const startOptions = recorder.handle.start.mock.calls.at(-1)?.[0]

    await act(async () => {
      startOptions.onSilence()
    })

    await waitFor(() => expect(onTranscribeAudio).toHaveBeenCalled())
    expect(onSubmit).toHaveBeenCalledWith('hello hermes')
    await waitFor(() => expect(recorder.handle.start).toHaveBeenCalledTimes(2))
    expect(result.current.status).toBe('listening')
  })

  it('lets the user interrupt a pending assistant response and returns to listening', async () => {
    const consumePendingResponse = vi.fn()

    const { result } = renderHook(() =>
      useVoiceConversation({
        busy: false,
        consumePendingResponse,
        enabled: true,
        onSubmit: vi.fn().mockResolvedValue(undefined),
        onTranscribeAudio: vi.fn().mockResolvedValue('interrupt test'),
        pendingResponse: () => ({ id: 'assistant-1', pending: true, text: 'partial assistant words' })
      })
    )

    await act(async () => {
      await result.current.start()
    })

    const startOptions = recorder.handle.start.mock.calls.at(-1)?.[0]

    await act(async () => {
      startOptions.onSilence()
    })

    await waitFor(() => expect(result.current.status).toBe('thinking'))

    await act(async () => {
      await result.current.interruptResponse()
    })

    expect(playback.stopVoicePlayback).toHaveBeenCalled()
    expect(consumePendingResponse).toHaveBeenCalled()
    await waitFor(() => expect(result.current.status).toBe('listening'))

    expect(recorder.handle.start).toHaveBeenCalledTimes(2)
  })

  it('cancels a busy backend turn before re-listening from thinking state', async () => {
    const onCancel = vi.fn().mockResolvedValue(undefined)
    const consumePendingResponse = vi.fn()

    const { result, rerender } = renderHook(
      ({ busy }) =>
        useVoiceConversation({
          busy,
          consumePendingResponse,
          enabled: true,
          onCancel,
          onSubmit: vi.fn().mockResolvedValue(undefined),
          onTranscribeAudio: vi.fn().mockResolvedValue('interrupt test'),
          pendingResponse: () => ({ id: 'assistant-1', pending: true, text: 'partial assistant words' })
        }),
      { initialProps: { busy: false } }
    )

    await act(async () => {
      await result.current.start()
    })

    const startOptions = recorder.handle.start.mock.calls.at(-1)?.[0]

    await act(async () => {
      startOptions.onSilence()
    })

    await waitFor(() => expect(result.current.status).toBe('thinking'))

    rerender({ busy: true })

    await act(async () => {
      await result.current.interruptResponse()
    })

    expect(onCancel).toHaveBeenCalledTimes(1)
    expect(playback.stopVoicePlayback).toHaveBeenCalled()
    expect(consumePendingResponse).toHaveBeenCalled()
    expect(result.current.status).toBe('idle')
  })
})
