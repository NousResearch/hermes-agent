import { act, cleanup, renderHook } from '@testing-library/react'
import type * as Nanostores from 'nanostores'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { VoiceCallTargetProvider } from '../voice-call-target'

import { useComposerVoice } from './use-composer-voice'
import type { useVoiceLiveConversation } from './use-voice-live-conversation'

const mocks = vi.hoisted(() => ({
  clearDraft: vi.fn(),
  conversationSubmits: [] as Array<(text: string) => Promise<void>>,
  liveOptions: [] as Array<Parameters<typeof useVoiceLiveConversation>[0]>,
  onSubmit: vi.fn(async () => true),
  resetBrowseState: vi.fn(),
  triggerHaptic: vi.fn()
}))

vi.mock('@nanostores/react', () => ({
  useStore: (store: unknown) => store
}))

vi.mock('nanostores', async () => {
  const actual = await vi.importActual<typeof Nanostores>('nanostores')

  return { ...actual, computed: (deps: unknown, fn: unknown) => fn }
})

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
  onComposerDictationRequest: vi.fn(() => () => undefined),
  onComposerVoiceToggleRequest: vi.fn(() => () => undefined)
}))

vi.mock('../scope', () => ({
  useComposerScope: () => ({ $messages: { get: () => [] } }),
  useComposerSurfaceId: () => null
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

vi.mock('./use-voice-live-conversation', () => ({
  useVoiceLiveConversation: (options: Parameters<typeof useVoiceLiveConversation>[0]) => {
    mocks.liveOptions.push(options)

    return { end: vi.fn(), start: vi.fn(), status: 'idle' }
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
    mocks.liveOptions.length = 0
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

    expect(mocks.onSubmit).toHaveBeenCalledWith('change direction', { voiceTurn: true })
    expect(mocks.clearDraft).toHaveBeenCalledTimes(1)
    expect(mocks.resetBrowseState).toHaveBeenCalledWith('session-1')
    expect(mocks.triggerHaptic).toHaveBeenCalledWith('submit')

    hook.rerender({ busy: true })
    await act(async () => retainedSubmit('do not double-submit'))

    expect(mocks.onSubmit).toHaveBeenCalledTimes(1)
    expect(mocks.clearDraft).toHaveBeenCalledTimes(1)
  })

  it('passes the stored call target and stable submission ID to the ordinary prompt path', async () => {
    renderHook(() =>
      useComposerVoice({
        busy: false,
        clearDraft: mocks.clearDraft,
        disabled: false,
        focusInput: vi.fn(),
        insertText: vi.fn(),
        maxRecordingSeconds: 60,
        onSubmit: mocks.onSubmit,
        onTranscribeAudio: vi.fn(async () => ''),
        sessionId: 'runtime-1',
        target: 'tile:test'
      }),
      {
        wrapper: ({ children }) => (
          <VoiceCallTargetProvider value={{ routeSessionId: 'stored-1', storedSessionId: 'stored-1' }}>
            {children}
          </VoiceCallTargetProvider>
        )
      }
    )

    const options = mocks.liveOptions.at(-1)!

    expect(options.storedSessionId).toBe('stored-1')
    await act(async () => options.onSubmit('Check the weather.', 'Recent voice context', 'stable-id', 'stored-1'))
    expect(mocks.onSubmit).toHaveBeenCalledWith('Check the weather.', {
      sessionId: 'runtime-1',
      storedSessionId: 'stored-1',
      submission_id: 'stable-id',
      surface: 'voice-live',
      voiceContext: 'Recent voice context'
    })
  })
})
