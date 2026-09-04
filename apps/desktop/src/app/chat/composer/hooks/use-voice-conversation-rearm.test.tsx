import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $voicePlayback } from '@/store/voice-playback'

import { useVoiceConversation } from './use-voice-conversation'

const mocks = vi.hoisted(() => {
  let deferStreamStart = false
  let onSilence: null | (() => void) = null
  let resolveStreamStart: null | (() => void) = null
  let resolveSpeech: null | ((outcome: 'done' | 'fallback') => void) = null
  let streamAvailable = true

  const stopVoicePlayback = vi.fn(() => {
    const current = $voicePlayback.get()
    $voicePlayback.set({ ...current, sequence: current.sequence + 1, status: 'idle' })
  })

  const playSpeechText = vi.fn(() => {
    stopVoicePlayback()

    return Promise.resolve(true)
  })

  const handle = {
    cancel: vi.fn(),
    start: vi.fn(async (options: { onSilence: () => void }) => {
      onSilence = options.onSilence

      return true
    }),
    stop: vi.fn(async () => ({
      audio: new Blob(['voice'], { type: 'audio/webm' }),
      heardSpeech: true
    }))
  }

  return {
    continueStreamStart() {
      resolveStreamStart?.()
      resolveStreamStart = null
    },
    deferStreamStart() {
      deferStreamStart = true
    },
    finishSpeech(outcome: 'done' | 'fallback') {
      resolveSpeech?.(outcome)
    },
    handle,
    playSpeechText,
    resetSpeechMocks() {
      deferStreamStart = false
      resolveStreamStart = null
      resolveSpeech = null
      streamAvailable = true
    },
    startSpeechStream: vi.fn(async () => {
      if (deferStreamStart) {
        await new Promise<void>(resolve => {
          resolveStreamStart = resolve
        })
      }

      if (!streamAvailable) {
        return null
      }

      const current = $voicePlayback.get()
      $voicePlayback.set({ ...current, sequence: current.sequence + 1, status: 'preparing' })

      return {
        append: vi.fn(),
        done: new Promise<'done' | 'fallback'>(resolve => {
          resolveSpeech = resolve
        }),
        finish: vi.fn()
      }
    }),
    stopVoicePlayback,
    triggerSilence() {
      onSilence?.()
    },
    useFallbackSpeech() {
      streamAvailable = false
    }
  }
})

vi.mock('./use-mic-recorder', () => ({
  useMicRecorder: () => ({ handle: mocks.handle, level: 0 })
}))

vi.mock('@/lib/voice-barge-in', () => ({
  monitorSpeechDuringPlayback: () => vi.fn()
}))

vi.mock('@/lib/voice-playback', () => ({
  markVoicePlaybackInterrupted: vi.fn(),
  playSpeechText: mocks.playSpeechText,
  startSpeechStream: mocks.startSpeechStream,
  stopVoicePlayback: mocks.stopVoicePlayback
}))

vi.mock('@/lib/thinking-sound', () => ({
  startThinkingSound: vi.fn(),
  stopThinkingSound: vi.fn()
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
          configureSpeechToText: '',
          couldNotStartSession: '',
          microphoneFailed: '',
          playbackFailed: '',
          transcriptionFailed: '',
          unavailable: ''
        }
      }
    }
  })
}))

function renderRearmConversation(responseId: string, responseText: string) {
  let response: null | { id: string; pending: boolean; text: string } = null

  return renderHook(
    ({ enabled }) =>
      useVoiceConversation({
        busy: false,
        consumePendingResponse: vi.fn(),
        enabled,
        onSubmit: async () => {
          response = { id: responseId, pending: false, text: responseText }
        },
        onTranscribeAudio: async () => 'Hello',
        pendingResponse: () => response
      }),
    { initialProps: { enabled: false } }
  )
}

async function beginReply(hook: ReturnType<typeof renderRearmConversation>) {
  hook.rerender({ enabled: true })
  await waitFor(() => expect(mocks.handle.start).toHaveBeenCalledTimes(1))

  await act(async () => {
    mocks.triggerSilence()
  })
}

describe('useVoiceConversation playback rearm', () => {
  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
    mocks.resetSpeechMocks()
    $voicePlayback.set({
      audioElement: null,
      messageId: null,
      sequence: 0,
      source: null,
      status: 'idle'
    })
  })

  it('re-arms the microphone after normal streaming playback completes', async () => {
    $voicePlayback.set({
      audioElement: null,
      messageId: null,
      sequence: 7,
      source: null,
      status: 'idle'
    })
    const hook = renderRearmConversation('reply-1', 'Hello back')

    await beginReply(hook)
    await waitFor(() => expect(mocks.startSpeechStream).toHaveBeenCalled())
    expect($voicePlayback.get().sequence).toBeGreaterThan(7)

    await act(async () => {
      mocks.finishSpeech('done')
    })

    await waitFor(() => expect(mocks.handle.start).toHaveBeenCalledTimes(2))
    expect(hook.result.current.status).toBe('listening')
  })

  it('honors Stop while streaming playback is still preparing', async () => {
    mocks.deferStreamStart()
    const hook = renderRearmConversation('reply-preparing', 'Do not play this')

    await beginReply(hook)
    await waitFor(() => expect(mocks.startSpeechStream).toHaveBeenCalled())

    mocks.stopVoicePlayback()
    await act(async () => {
      mocks.continueStreamStart()
    })

    await waitFor(() => expect(hook.result.current.status).toBe('idle'))
    expect(mocks.stopVoicePlayback).toHaveBeenCalledTimes(2)
    expect(mocks.handle.start).toHaveBeenCalledTimes(1)
  })

  it('does not start fallback playback after Stop during stream discovery', async () => {
    mocks.deferStreamStart()
    mocks.useFallbackSpeech()
    const hook = renderRearmConversation('reply-no-stream', 'Do not fall back')

    await beginReply(hook)
    await waitFor(() => expect(mocks.startSpeechStream).toHaveBeenCalled())

    mocks.stopVoicePlayback()
    await act(async () => {
      mocks.continueStreamStart()
    })

    await waitFor(() => expect(hook.result.current.status).toBe('idle'))
    expect(mocks.playSpeechText).not.toHaveBeenCalled()
    expect(mocks.handle.start).toHaveBeenCalledTimes(1)
  })

  it('does not re-arm after an external Stop during streaming playback', async () => {
    const hook = renderRearmConversation('reply-stopped', 'Playing now')

    await beginReply(hook)
    await waitFor(() => expect(mocks.startSpeechStream).toHaveBeenCalled())

    mocks.stopVoicePlayback()
    await act(async () => {
      mocks.finishSpeech('done')
    })

    await waitFor(() => expect(hook.result.current.status).toBe('idle'))
    expect(mocks.handle.start).toHaveBeenCalledTimes(1)
  })

  it('re-arms the microphone after normal fallback playback completes', async () => {
    mocks.useFallbackSpeech()
    const hook = renderRearmConversation('reply-fallback', 'Fallback reply')

    await beginReply(hook)

    await waitFor(() =>
      expect(mocks.playSpeechText).toHaveBeenCalledWith('Fallback reply', {
        source: 'voice-conversation'
      })
    )
    await waitFor(() => expect(mocks.handle.start).toHaveBeenCalledTimes(2))
    expect(hook.result.current.status).toBe('listening')
  })
})
