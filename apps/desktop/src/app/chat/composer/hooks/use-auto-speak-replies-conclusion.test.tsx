import { act, cleanup, renderHook } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { assistantTextPart, type ChatMessage, chatMessageText } from '@/lib/chat-messages'
import { clearSpokenRepliesForTests, markAssistantIdSpoken, resolveSpokenReply } from '@/lib/spoken-reply'
import { playSpeechText } from '@/lib/voice-playback'
import { setVoicePlaybackState } from '@/store/voice-playback'
import { $autoSpeakReplies, $ttsConclusionGraceMs, $ttsConclusionOnly } from '@/store/voice-prefs'

import { ComposerScopeProvider, MAIN_COMPOSER_SCOPE } from '../scope'

import { useAutoSpeakReplies } from './use-auto-speak-replies'

vi.mock('@/lib/voice-playback', () => ({
  playSpeechText: vi.fn(async () => true)
}))

const SESSION_ID = 'session-under-test'
const IDLE_STATE = { audioElement: null, messageId: null, sequence: 0, source: null, status: 'idle' as const }

function assistantMessage(id: string, text: string): ChatMessage {
  return { id, parts: [assistantTextPart(text)], role: 'assistant' }
}

describe('useAutoSpeakReplies — conclusion-only debounce (#107056)', () => {
  afterEach(() => {
    cleanup()
    clearSpokenRepliesForTests()
    $autoSpeakReplies.set(false)
    $ttsConclusionOnly.set(false)
    $ttsConclusionGraceMs.set(1500)
    setVoicePlaybackState({ ...IDLE_STATE })
    vi.clearAllMocks()
  })

  const wireMessages = () => {
    const $messages = atom<ChatMessage[]>([])

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

    return { $messages, pendingReply, markSpoken }
  }

  const render = ($messages: ReturnType<typeof wireMessages>['$messages'], pendingReply: () => { id: string; pending: boolean; text: string } | null, markSpoken: () => void) =>
    renderHook(
      () =>
        useAutoSpeakReplies({
          conversationActive: false,
          failureLabel: 'read-aloud failed',
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

  it('speaks only the final reply of a burst after the grace window', async () => {
    vi.useFakeTimers()
    $autoSpeakReplies.set(true)
    $ttsConclusionOnly.set(true)
    $ttsConclusionGraceMs.set(1500)

    const { $messages, pendingReply, markSpoken } = wireMessages()
    render($messages, pendingReply, markSpoken)

    await act(async () => {
      $messages.set([assistantMessage('interim-1', 'thinking about it')])
    })
    await act(async () => {
      await vi.advanceTimersByTimeAsync(700)
    })
    await act(async () => {
      $messages.set([assistantMessage('interim-1', 'thinking about it'), assistantMessage('interim-2', 'progress')])
    })
    await act(async () => {
      await vi.advanceTimersByTimeAsync(700)
    })

    // Interim replies within the window never fire speech.
    expect(playSpeechText).not.toHaveBeenCalled()

    await act(async () => {
      $messages.set([
        assistantMessage('interim-1', 'thinking about it'),
        assistantMessage('interim-2', 'progress'),
        assistantMessage('final-1', 'the answer')
      ])
    })

    await act(async () => {
      vi.advanceTimersByTime(1499)
    })
    expect(playSpeechText).not.toHaveBeenCalled()

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1)
    })

    expect(playSpeechText).toHaveBeenCalledTimes(1)
    expect(vi.mocked(playSpeechText).mock.calls[0]?.[0]).toBe('the answer')

    vi.useRealTimers()
  })

  it('speaks every reply immediately when conclusion-only is off', async () => {
    vi.useFakeTimers()
    $autoSpeakReplies.set(true)
    $ttsConclusionOnly.set(false)

    const { $messages, pendingReply, markSpoken } = wireMessages()
    render($messages, pendingReply, markSpoken)

    await act(async () => {
      $messages.set([assistantMessage('reply-1', 'first')])
      await vi.advanceTimersByTimeAsync(0)
    })

    expect(playSpeechText).toHaveBeenCalledTimes(1)
    expect(vi.mocked(playSpeechText).mock.calls[0]?.[0]).toBe('first')

    vi.useRealTimers()
  })

  it('speaks a reply arriving after the mode flips off without waiting', async () => {
    vi.useFakeTimers()
    $autoSpeakReplies.set(true)
    $ttsConclusionOnly.set(true)
    $ttsConclusionGraceMs.set(1500)

    const { $messages, pendingReply, markSpoken } = wireMessages()
    render($messages, pendingReply, markSpoken)

    await act(async () => {
      $messages.set([assistantMessage('reply-1', 'hello')])
    })

    // Pending conclusion timer from conclusion-only mode still fires with the
    // mode already off: speakReply is mode-independent by design, so the reply
    // is not lost to the flip.
    await act(async () => {
      $ttsConclusionOnly.set(false)
      await vi.advanceTimersByTimeAsync(1500)
    })

    expect(playSpeechText).toHaveBeenCalledTimes(1)
    expect(vi.mocked(playSpeechText).mock.calls[0]?.[0]).toBe('hello')

    // New replies after the flip speak immediately, no grace window.
    await act(async () => {
      $messages.set([assistantMessage('reply-1', 'hello'), assistantMessage('reply-2', 'second')])
      await vi.advanceTimersByTimeAsync(0)
    })

    expect(playSpeechText).toHaveBeenCalledTimes(2)
    expect(vi.mocked(playSpeechText).mock.calls[1]?.[0]).toBe('second')

    vi.useRealTimers()
  })

  it('does not fire pending speech after unmount', async () => {
    vi.useFakeTimers()
    $autoSpeakReplies.set(true)
    $ttsConclusionOnly.set(true)
    $ttsConclusionGraceMs.set(1500)

    const { $messages, pendingReply, markSpoken } = wireMessages()
    const { unmount } = render($messages, pendingReply, markSpoken)

    await act(async () => {
      $messages.set([assistantMessage('reply-1', 'hello')])
    })

    unmount()

    await act(async () => {
      await vi.advanceTimersByTimeAsync(5000)
    })

    expect(playSpeechText).not.toHaveBeenCalled()

    vi.useRealTimers()
  })
})
