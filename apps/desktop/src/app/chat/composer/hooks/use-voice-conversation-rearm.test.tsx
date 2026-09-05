import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $voicePlayback } from '@/store/voice-playback'

import { useVoiceConversation } from './use-voice-conversation'

const mocks = vi.hoisted(() => {
  let deferStreamStart = false
  let deferFallbackPlayback = false
  let onSilence: null | (() => void) = null
  let resolveFallbackPlayback: null | ((played: boolean) => void) = null
  let resolveStreamStart: null | (() => void) = null
  let resolveSpeech: null | ((outcome: 'done' | 'fallback') => void) = null
  let streamAvailable = true

  const stopVoicePlayback = vi.fn(() => {
    const current = $voicePlayback.get()
    $voicePlayback.set({ ...current, sequence: current.sequence + 1, status: 'idle' })
  })

  const playSpeechText = vi.fn(() => {
    stopVoicePlayback()

    if (deferFallbackPlayback) {
      return new Promise<boolean>(resolve => {
        resolveFallbackPlayback = resolve
      })
    }

    return Promise.resolve(true)
  })

  const handle = {
    cancel: vi.fn(),
    start: vi.fn(async (options: { onSilence: () => void }) => {
      onSilence = options.onSilence
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
    deferFallbackPlayback() {
      deferFallbackPlayback = true
    },
    deferStreamStart() {
      deferStreamStart = true
    },
    finishSpeech(outcome: 'done' | 'fallback') {
      resolveSpeech?.(outcome)
    },
    finishFallbackPlayback() {
      resolveFallbackPlayback?.(true)
      resolveFallbackPlayback = null
    },
    handle,
    playSpeechText,
    resetSpeechMocks() {
      deferFallbackPlayback = false
      deferStreamStart = false
      resolveFallbackPlayback = null
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

function renderIncrementalFallbackConversation() {
  let response: null | { id: string; pending: boolean; text: string } = null

  const hook = renderHook(
    ({ enabled }) =>
      useVoiceConversation({
        busy: false,
        consumePendingResponse: vi.fn(),
        enabled,
        onSubmit: async () => {
          response = { id: 'reply-edge', pending: true, text: 'The first sentence is ready. ' }
        },
        onTranscribeAudio: async () => 'Hello',
        pendingResponse: () => response
      }),
    { initialProps: { enabled: false } }
  )

  return {
    finishResponse() {
      response = {
        id: 'reply-edge',
        pending: false,
        text: 'The first sentence is ready. The second sentence is ready.'
      }
    },
    hook
  }
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

  it('speaks completed fallback sentences before the response finishes', async () => {
    mocks.useFallbackSpeech()
    const { finishResponse, hook } = renderIncrementalFallbackConversation()

    await beginReply(hook)

    await waitFor(() =>
      expect(mocks.playSpeechText).toHaveBeenCalledWith('The first sentence is ready.', {
        source: 'voice-conversation'
      })
    )
    expect(mocks.handle.start).toHaveBeenCalledTimes(1)

    finishResponse()

    await waitFor(() =>
      expect(mocks.playSpeechText).toHaveBeenCalledWith('The second sentence is ready.', {
        source: 'voice-conversation'
      })
    )
    await waitFor(() => expect(mocks.handle.start).toHaveBeenCalledTimes(2))
    expect(hook.result.current.status).toBe('listening')
  })

  it('does not play the next fallback sentence or re-arm after Stop', async () => {
    mocks.useFallbackSpeech()
    mocks.deferFallbackPlayback()
    const { finishResponse, hook } = renderIncrementalFallbackConversation()

    await beginReply(hook)
    await waitFor(() => expect(mocks.playSpeechText).toHaveBeenCalledTimes(1))
    finishResponse()

    mocks.stopVoicePlayback()
    await act(async () => {
      mocks.finishFallbackPlayback()
    })

    await waitFor(() => expect(hook.result.current.status).toBe('idle'))
    expect(mocks.playSpeechText).toHaveBeenCalledTimes(1)
    expect(mocks.handle.start).toHaveBeenCalledTimes(1)
  })

  it('honors Stop while waiting for the next fallback sentence', async () => {
    mocks.useFallbackSpeech()
    const { finishResponse, hook } = renderIncrementalFallbackConversation()

    await beginReply(hook)
    await waitFor(() => expect(mocks.playSpeechText).toHaveBeenCalledTimes(1))

    mocks.stopVoicePlayback()
    await waitFor(() => expect(hook.result.current.status).toBe('idle'))
    finishResponse()

    expect(mocks.playSpeechText).toHaveBeenCalledTimes(1)
    expect(mocks.handle.start).toHaveBeenCalledTimes(1)
  })
})
