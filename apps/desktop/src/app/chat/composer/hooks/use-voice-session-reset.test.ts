import { cleanup, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { stopVoicePlayback } from '@/lib/voice-playback'

import { useComposerVoice } from './use-composer-voice'
import { useEndVoiceOnSessionSwitch, useStopVoicePlaybackOnUnmount } from './use-voice-session-reset'

const voiceMocks = vi.hoisted(() => ({
  conversation: { end: vi.fn(), status: 'idle' },
  recorder: {
    cancel: vi.fn(),
    dictate: vi.fn(),
    voiceActivityState: { elapsedSeconds: 3, level: 0.5, status: 'recording' },
    voiceStatus: 'recording'
  }
}))

vi.mock('@/lib/voice-playback', () => ({
  stopVoicePlayback: vi.fn()
}))

vi.mock('@nanostores/react', () => ({ useStore: () => null }))
vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      assistant: { thread: { readAloudFailed: '' } },
      notifications: { voice: { sayStopToEnd: () => '' } },
      settings: { config: { autosaveFailed: '' } }
    }
  })
}))
vi.mock('@/lib/chat-messages', () => ({ chatMessageText: () => '', collectUnspokenTurnSpeech: () => [] }))
vi.mock('@/lib/haptics', () => ({ triggerHaptic: vi.fn() }))
vi.mock('@/lib/wake-indicator', () => ({ clearWakeIndicator: vi.fn(), syncWakeIndicatorWithVoice: () => false }))
vi.mock('@/store/composer', () => ({ $voiceConversationStartRequest: {}, takeVoiceConversationStart: () => false }))
vi.mock('@/store/composer-input-history', () => ({ resetBrowseState: vi.fn() }))
vi.mock('@/store/gateway', () => ({ $gateway: { get: () => null } }))
vi.mock('@/store/notifications', () => ({ notify: vi.fn(), notifyError: vi.fn() }))
vi.mock('@/store/voice-prefs', () => ({
  $autoSpeakReplies: { get: () => false },
  $voiceStopPhrase: { get: () => null },
  setAutoSpeakReplies: vi.fn()
}))
vi.mock('@/store/wake-word', () => ({ resumeWakeAfterVoice: vi.fn() }))
vi.mock('../focus', () => ({ onComposerVoiceToggleRequest: () => () => undefined }))
vi.mock('../scope', () => ({ useComposerScope: () => ({ $messages: { get: () => [] } }) }))
vi.mock('./use-auto-speak-replies', () => ({ useAutoSpeakReplies: vi.fn() }))
vi.mock('./use-voice-conversation', () => ({ useVoiceConversation: () => voiceMocks.conversation }))
vi.mock('./use-voice-recorder', () => ({ useVoiceRecorder: () => voiceMocks.recorder }))

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

function setup(initial: string | null | undefined) {
  const onSwitch = vi.fn()
  const view = renderHook(({ id }: { id: string | null | undefined }) => useEndVoiceOnSessionSwitch(id, onSwitch), {
    initialProps: { id: initial }
  })

  return { onSwitch, ...view }
}

describe('useEndVoiceOnSessionSwitch', () => {
  it('does not fire on initial mount', () => {
    const { onSwitch } = setup('session-a')

    expect(onSwitch).not.toHaveBeenCalled()
  })

  it('does not fire when the session id is unchanged', () => {
    const { onSwitch, rerender } = setup('session-a')

    rerender({ id: 'session-a' })
    rerender({ id: 'session-a' })

    expect(onSwitch).not.toHaveBeenCalled()
  })

  it('fires once for each switch between real sessions', () => {
    const { onSwitch, rerender } = setup('session-a')

    rerender({ id: 'session-b' })
    rerender({ id: 'session-c' })
    rerender({ id: 'session-a' })

    expect(onSwitch).toHaveBeenCalledTimes(3)
  })

  it('does not fire on the first no-session to session persist', () => {
    const { onSwitch, rerender } = setup(null)

    rerender({ id: 'session-a' })

    expect(onSwitch).not.toHaveBeenCalled()
  })

  it('treats undefined and null as the same no-session state', () => {
    const { onSwitch, rerender } = setup(undefined)

    rerender({ id: null })
    rerender({ id: 'session-a' })

    expect(onSwitch).not.toHaveBeenCalled()
  })

  it('fires when leaving a session', () => {
    const { onSwitch, rerender } = setup('session-a')

    rerender({ id: null })

    expect(onSwitch).toHaveBeenCalledTimes(1)
  })

  it('uses the latest switch callback', () => {
    const first = vi.fn()
    const second = vi.fn()
    const { rerender } = renderHook(
      ({ id, cb }: { cb: () => void; id: string | null }) => useEndVoiceOnSessionSwitch(id, cb),
      { initialProps: { cb: first, id: 'session-a' as string | null } }
    )

    rerender({ cb: second, id: 'session-b' })

    expect(first).not.toHaveBeenCalled()
    expect(second).toHaveBeenCalledTimes(1)
  })
})

describe('useStopVoicePlaybackOnUnmount', () => {
  it('does not stop playback while mounted', () => {
    renderHook(() => useStopVoicePlaybackOnUnmount())

    expect(stopVoicePlayback).not.toHaveBeenCalled()
  })

  it('stops global voice playback on unmount', () => {
    const { unmount } = renderHook(() => useStopVoicePlaybackOnUnmount())

    unmount()

    expect(stopVoicePlayback).toHaveBeenCalledTimes(1)
  })
})

describe('useComposerVoice session switches', () => {
  it('cancels active dictation and ends the voice conversation', () => {
    const { rerender } = renderHook(
      ({ sessionId }: { sessionId: string }) =>
        useComposerVoice({
          busy: false,
          clearDraft: vi.fn(),
          disabled: false,
          focusInput: vi.fn(),
          insertText: vi.fn(),
          maxRecordingSeconds: 60,
          onSubmit: vi.fn(),
          onTranscribeAudio: vi.fn(),
          sessionId,
          target: 'main'
        }),
      { initialProps: { sessionId: 'session-a' } }
    )

    rerender({ sessionId: 'session-b' })

    expect(voiceMocks.recorder.cancel).toHaveBeenCalledTimes(1)
    expect(voiceMocks.conversation.end).toHaveBeenCalledTimes(1)
  })
})
