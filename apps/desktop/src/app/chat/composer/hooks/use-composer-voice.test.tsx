import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { useComposerVoice } from './use-composer-voice'

const mocks = vi.hoisted(() => ({
  clearDraft: vi.fn(),
  conversationSubmits: [] as Array<(text: string) => Promise<void>>,
  onSubmit: vi.fn(async () => true),
  resetBrowseState: vi.fn(),
  triggerHaptic: vi.fn()
}))

vi.mock('@nanostores/react', () => ({
  useStore: (store: { get: () => unknown }) => store.get()
}))

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      assistant: { thread: { readAloudFailed: '' } },
      notifications: { voice: { sayStopToEnd: () => '' } },
      settings: { config: { autosaveFailed: '' } }
    }
  })
}))

vi.mock('@/lib/haptics', () => ({
  triggerHaptic: mocks.triggerHaptic
}))

vi.mock('@/lib/spoken-reply', () => ({
  adoptSpokenReplySession: vi.fn(),
  markAssistantIdSpoken: vi.fn(),
  resolveSpokenReply: vi.fn(() => null)
}))

vi.mock('@/lib/tts-lease', () => ({
  CONVERSATION_LEASE: 'conversation',
  READ_ALOUD_LEASE: 'read-aloud',
  syncTtsLease: vi.fn(async () => undefined)
}))

vi.mock('@/lib/wake-indicator', () => ({
  clearWakeIndicator: vi.fn(),
  syncWakeIndicatorWithVoice: vi.fn(() => false)
}))

vi.mock('@/store/composer', () => ({
  $voiceConversationStartRequest: { get: () => null },
  takeVoiceConversationStart: vi.fn(() => false)
}))

vi.mock('@/store/composer-input-history', () => ({
  resetBrowseState: mocks.resetBrowseState
}))

vi.mock('@/store/gateway', () => ({
  $gateway: { get: () => null }
}))

vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: vi.fn()
}))

vi.mock('@/store/voice-prefs', () => ({
  $autoSpeakReplies: { get: () => false },
  $voiceStopPhrase: { get: () => null },
  setAutoSpeakReplies: vi.fn(async () => undefined)
}))

vi.mock('@/store/wake-word', () => ({
  resumeWakeAfterVoice: vi.fn(async () => undefined)
}))

vi.mock('../focus', () => ({
  onComposerVoiceToggleRequest: vi.fn(() => () => undefined)
}))

vi.mock('../scope', () => ({
  useComposerScope: () => ({ $messages: { get: () => [] } })
}))

vi.mock('./use-auto-speak-replies', () => ({
  useAutoSpeakReplies: vi.fn()
}))

vi.mock('./use-voice-conversation', () => ({
  useVoiceConversation: ({ onSubmit }: { onSubmit: (text: string) => Promise<void> }) => {
    mocks.conversationSubmits.push(onSubmit)

    return {
      end: vi.fn(async () => undefined),
      level: 0,
      muted: false,
      start: vi.fn(async () => undefined),
      status: 'idle',
      stopTurn: vi.fn(),
      toggleMute: vi.fn()
    }
  }
}))

vi.mock('./use-voice-recorder', () => ({
  useVoiceRecorder: () => ({
    dictate: vi.fn(),
    voiceActivityState: { elapsedSeconds: 0, level: 0, status: 'idle' },
    voiceStatus: 'idle'
  })
}))

describe('useComposerVoice voice submission', () => {
  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
    mocks.conversationSubmits.length = 0
  })

  it('uses live busy state when a retained voice callback submits after barge-in', async () => {
    const hook = renderHook(
      ({ busy }) =>
        useComposerVoice({
          busy,
          clearDraft: mocks.clearDraft,
          disabled: false,
          focusInput: vi.fn(),
          insertText: vi.fn(),
          maxRecordingSeconds: 60,
          onSubmit: mocks.onSubmit,
          onTranscribeAudio: vi.fn(async () => ''),
          sessionId: 'session-1',
          target: 'tile:test'
        }),
      { initialProps: { busy: true } }
    )

    const retainedSubmit = mocks.conversationSubmits[0]

    hook.rerender({ busy: false })
    await act(async () => retainedSubmit('change direction'))

    expect(mocks.onSubmit).toHaveBeenCalledWith('change direction')
    expect(mocks.clearDraft).toHaveBeenCalledTimes(1)
    expect(mocks.resetBrowseState).toHaveBeenCalledWith('session-1')
    expect(mocks.triggerHaptic).toHaveBeenCalledWith('submit')

    hook.rerender({ busy: true })
    await act(async () => retainedSubmit('do not double-submit'))

    expect(mocks.onSubmit).toHaveBeenCalledTimes(1)
    expect(mocks.clearDraft).toHaveBeenCalledTimes(1)
  })
})
