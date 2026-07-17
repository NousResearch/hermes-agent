import { act, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { markActiveComposer, requestDictationToggle } from '../focus'

import { useComposerVoice } from './use-composer-voice'

const recorder = vi.hoisted(() => ({
  cancel: vi.fn(),
  dictate: vi.fn(),
  options: null as null | {
    onIdle?: () => void
    onTranscript: (text: string, submit: boolean) => void
  },
  status: 'idle' as 'idle' | 'recording' | 'transcribing'
}))

const conversation = vi.hoisted(() => ({
  end: vi.fn(),
  level: 0,
  muted: false,
  status: 'idle' as const,
  stopTurn: vi.fn(),
  toggleMute: vi.fn()
}))

vi.mock('./use-voice-recorder', () => ({
  useVoiceRecorder: (options: {
    onIdle?: () => void
    onTranscript: (text: string, submit: boolean) => void
  }) => {
    recorder.options = options

    return {
      cancel: recorder.cancel,
      dictate: recorder.dictate,
      voiceActivityState: { elapsedSeconds: 0, level: 0, status: recorder.status },
      voiceStatus: recorder.status
    }
  }
}))

vi.mock('./use-voice-conversation', () => ({ useVoiceConversation: () => conversation }))
vi.mock('./use-auto-speak-replies', () => ({ useAutoSpeakReplies: () => undefined }))
vi.mock('@/lib/haptics', () => ({ triggerHaptic: vi.fn() }))
vi.mock('@/store/composer-input-history', () => ({ resetBrowseState: vi.fn() }))
vi.mock('@/store/notifications', () => ({ notifyError: vi.fn() }))
vi.mock('@/store/session', () => ({ $messages: { get: () => [] } }))
vi.mock('@/store/voice-prefs', () => ({
  $autoSpeakReplies: { get: () => false },
  setAutoSpeakReplies: vi.fn().mockResolvedValue(undefined)
}))
vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      assistant: { thread: { readAloudFailed: 'Read aloud failed' } },
      settings: { config: { autosaveFailed: 'Autosave failed' } }
    }
  })
}))

beforeEach(() => {
  vi.useFakeTimers()
  vi.clearAllMocks()
  recorder.options = null
  recorder.status = 'idle'
  recorder.cancel.mockImplementation(() => recorder.options?.onIdle?.())
  markActiveComposer('main')
})

afterEach(() => {
  vi.useRealTimers()
})

function renderVoice(target = 'main', initialScopeKey = 'session-1') {
  const insertText = vi.fn()
  const submitDraft = vi.fn()

  const hook = renderHook(
    ({ scopeKey }) =>
      useComposerVoice({
        busy: false,
        clearDraft: vi.fn(),
        dictationEnabled: true,
        dictationScopeKey: scopeKey,
        disabled: false,
        focusInput: vi.fn(),
        insertText,
        maxRecordingSeconds: 60,
        onSubmit: vi.fn().mockResolvedValue(true),
        onTranscribeAudio: vi.fn().mockResolvedValue('spoken words'),
        sessionId: scopeKey,
        submitDraft,
        target
      }),
    { initialProps: { scopeKey: initialScopeKey } }
  )

  return { ...hook, insertText, submitDraft }
}

describe('useComposerVoice dictation hotkey', () => {
  it('pins the active composer through the second press, then submits', () => {
    const { insertText, submitDraft } = renderVoice()

    act(() => requestDictationToggle())
    act(() => vi.runOnlyPendingTimers())
    expect(recorder.dictate).toHaveBeenLastCalledWith(true)

    markActiveComposer('tile:other')
    act(() => requestDictationToggle())
    act(() => vi.runOnlyPendingTimers())
    expect(recorder.dictate).toHaveBeenCalledTimes(2)
    expect(recorder.dictate).toHaveBeenLastCalledWith(true)

    act(() => recorder.options?.onTranscript('spoken words', true))
    expect(insertText).toHaveBeenCalledWith('spoken words')
    expect(submitDraft).toHaveBeenCalledOnce()
    expect(insertText.mock.invocationCallOrder[0]).toBeLessThan(submitDraft.mock.invocationCallOrder[0])
  })

  it('ignores requests routed to another mounted composer', () => {
    renderVoice('tile:one')

    act(() => requestDictationToggle('main'))
    act(() => vi.runOnlyPendingTimers())

    expect(recorder.dictate).not.toHaveBeenCalled()
  })

  it('cancels and rejects a stale transcript when the session changes', () => {
    const { insertText, rerender, submitDraft } = renderVoice('main', 'session-a')

    act(() => requestDictationToggle('main'))
    act(() => vi.runOnlyPendingTimers())
    const staleTranscript = recorder.options?.onTranscript

    rerender({ scopeKey: 'session-b' })

    expect(recorder.cancel).toHaveBeenCalledOnce()
    act(() => staleTranscript?.('wrong session', true))
    expect(insertText).not.toHaveBeenCalled()
    expect(submitDraft).not.toHaveBeenCalled()
  })
})
