import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { assistantTextPart, type ChatMessage, chatMessageText } from '@/lib/chat-messages'
import { clearSpokenRepliesForTests, markAssistantIdSpoken, resolveSpokenReply } from '@/lib/spoken-reply'
import { playSpeechText, type SpeechStreamSession, startSpeechStream } from '@/lib/voice-playback'
import { $messages as $viewMessages, setActiveSessionId, setSelectedStoredSessionId } from '@/store/session'
import { clearAllSessionStates } from '@/store/session-states'
import { setVoicePlaybackState } from '@/store/voice-playback'
import { $autoSpeakReplies } from '@/store/voice-prefs'

import { ComposerScopeProvider, MAIN_COMPOSER_SCOPE } from '../scope'

import { clearOwnedSpeechTurnsForTests, useAutoSpeakReplies } from './use-auto-speak-replies'

vi.mock('@/lib/voice-playback', () => ({
  playSpeechText: vi.fn(),
  startSpeechStream: vi.fn()
}))

const SESSION_ID = 'session-under-test'
const IDLE_STATE = { audioElement: null, messageId: null, sequence: 0, source: null, status: 'idle' as const }

function assistantMessage(id: string, text: string, pending: boolean): ChatMessage {
  return { id, parts: [assistantTextPart(text)], pending, role: 'assistant' } as ChatMessage
}

function fakeSession(): SpeechStreamSession {
  let resolveDone: (v: 'done' | 'fallback') => void = () => {}

  return {
    append: vi.fn(),
    done: new Promise<'done' | 'fallback'>(resolve => { resolveDone = resolve }),
    finish: vi.fn(() => resolveDone('done'))
  }
}

function harness($messages: ReturnType<typeof atom<ChatMessage[]>>) {
  const pendingReply = () => {
    const messages = $messages.get()
    const last = messages.findLast(m => m.role === 'assistant' && !m.hidden)
    const spoken = resolveSpokenReply(SESSION_ID, messages)

    if (!last || last.id === spoken?.id) {
      return null
    }

    return { id: last.id, pending: Boolean(last.pending), text: chatMessageText(last) }
  }

  const markSpoken = () => {
    const messages = $messages.get()
    const last = messages.findLast(m => m.role === 'assistant' && !m.hidden)

    if (last) {
      markAssistantIdSpoken(SESSION_ID, messages, last.id)
    }
  }

  return { markSpoken, pendingReply }
}

describe('useAutoSpeakReplies — streamed read-aloud (voice-out starts on the first sentence)', () => {
  afterEach(() => {
    cleanup()
    clearSpokenRepliesForTests()
    clearOwnedSpeechTurnsForTests()
    $autoSpeakReplies.set(false)
    setVoicePlaybackState({ ...IDLE_STATE })
    clearAllSessionStates()
    vi.clearAllMocks()
  })

  it('opens a live session on a pending reply, appends only the growth, and never re-says it whole at settle', async () => {
    $autoSpeakReplies.set(true)
    const session = fakeSession()

    vi.mocked(startSpeechStream).mockResolvedValue(session)

    const $messages = atom<ChatMessage[]>([])
    const { markSpoken, pendingReply } = harness($messages)

    renderHook(
      () => useAutoSpeakReplies({
        conversationActive: false,
        failureLabel: 'test',
        markSpoken,
        pendingReply,
        sessionId: SESSION_ID
      }),
      {
        wrapper: ({ children }) => (
          <ComposerScopeProvider value={{ ...MAIN_COMPOSER_SCOPE, $messages }}>{children}</ComposerScopeProvider>
        )
      }
    )

    setActiveSessionId(SESSION_ID)
    setSelectedStoredSessionId(SESSION_ID)

    // First wake: a partial pending bubble opens the stream and carries its text.
    act(() => { $viewMessages.set([]); $messages.set([assistantMessage('m1', 'First sentence.', true)]) })
    await waitFor(() => expect(startSpeechStream).toHaveBeenCalledTimes(1))
    expect(session.append).toHaveBeenCalledWith('First sentence.')

    // Growth: only the new tail is appended — the server cuts at sentence seams.
    act(() => { $messages.set([assistantMessage('m1', 'First sentence. Second one lands.', true)]) })
    await waitFor(() => expect(session.append).toHaveBeenCalledWith(' Second one lands.'))
    expect(session.append).toHaveBeenCalledTimes(2)

    // Settle: seal the stream; the whole-text speak must NOT fire (no double).
    act(() => { $viewMessages.set([]); $messages.set([assistantMessage('m1', 'First sentence. Second one lands.', false)]) })
    await waitFor(() => expect(session.finish).toHaveBeenCalledTimes(1))
    expect(playSpeechText).not.toHaveBeenCalled()
    expect(session.append).toHaveBeenCalledTimes(2)
  })

  it('a provider that cannot stream still speaks the settled reply whole (old behavior)', async () => {
    $autoSpeakReplies.set(true)
    vi.mocked(startSpeechStream).mockResolvedValue(null)
    vi.mocked(playSpeechText).mockResolvedValue(true)

    const $messages = atom<ChatMessage[]>([])
    const { markSpoken, pendingReply } = harness($messages)

    renderHook(
      () => useAutoSpeakReplies({
        conversationActive: false,
        failureLabel: 'test',
        markSpoken,
        pendingReply,
        sessionId: SESSION_ID
      }),
      {
        wrapper: ({ children }) => (
          <ComposerScopeProvider value={{ ...MAIN_COMPOSER_SCOPE, $messages }}>{children}</ComposerScopeProvider>
        )
      }
    )

    setActiveSessionId(SESSION_ID)
    setSelectedStoredSessionId(SESSION_ID)

    act(() => { $viewMessages.set([]); $messages.set([assistantMessage('m2', 'Hello there.', true)]) })
    await waitFor(() => expect(startSpeechStream).toHaveBeenCalled())
    expect(playSpeechText).not.toHaveBeenCalled()

    act(() => { $viewMessages.set([]); $messages.set([assistantMessage('m2', 'Hello there.', false)]) })
    await waitFor(() => expect(playSpeechText).toHaveBeenCalledTimes(1))
    expect(vi.mocked(playSpeechText).mock.calls[0][0]).toBe('Hello there.')
  })
})
