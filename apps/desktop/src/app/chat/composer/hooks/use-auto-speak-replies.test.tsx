import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  assistantTextPart,
  type ChatMessage,
  type ChatMessagePart,
  chatMessageSpeechText,
  chatMessageText
} from '@/lib/chat-messages'
import {
  assistantTurnKey,
  clearSpokenRepliesForTests,
  markAssistantIdSpoken,
  markTurnSpokenThrough,
  nextUnspokenTurnPiece,
  resolveSpokenReply
} from '@/lib/spoken-reply'
import { playSpeechText } from '@/lib/voice-playback'
import { $voicePlayback, setVoicePlaybackState } from '@/store/voice-playback'
import { $autoSpeakReplies } from '@/store/voice-prefs'

import { ComposerScopeProvider, MAIN_COMPOSER_SCOPE } from '../scope'

import { useAutoSpeakReplies } from './use-auto-speak-replies'

vi.mock('@/lib/voice-playback', () => ({
  playSpeechText: vi.fn()
}))

const SESSION_ID = 'session-under-test'
const DESKTOP_BRIDGE = window.hermesDesktop
const IDLE_STATE = { audioElement: null, messageId: null, sequence: 0, source: null, status: 'idle' as const }

function assistantMessage(id: string, text: string): ChatMessage {
  return { id, parts: [assistantTextPart(text)], role: 'assistant' }
}

const toolCall = (name: string) =>
  ({ args: {}, toolCallId: name, toolName: name, type: 'tool-call' }) as ChatMessagePart

// #93515 — Edge TTS has no chunked-PCM API, so the WS attempt in
// playSpeechText's fallback ladder settles 'fallback' before any audio plays
// and the client retries over the POST endpoint. While that POST round-trip
// is in flight, the backend can rewrite the just-completed reply's renderer
// id (`assistant-stream-*`) to its durable id. The issue claims
// `resolveSpokenReply()` fails to follow that rewrite and the reply gets
// spoken a second time once `$voicePlayback` goes idle.
describe('useAutoSpeakReplies — Edge TTS fallback chain (#93515)', () => {
  afterEach(() => {
    cleanup()
    clearSpokenRepliesForTests()
    $autoSpeakReplies.set(false)
    setVoicePlaybackState({ ...IDLE_STATE })
    window.hermesDesktop = DESKTOP_BRIDGE
    vi.clearAllMocks()
  })

  it('does not re-speak the reply once playback goes idle after an id rewrite mid-fallback', async () => {
    $autoSpeakReplies.set(true)

    const $messages = atom<ChatMessage[]>([])

    // The exact pendingReply/markSpoken contract use-composer-voice.ts wires
    // up for this hook, backed by the real ordinal-anchored dedupe.
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

    let settleFallback: (() => void) | null = null

    vi.mocked(playSpeechText).mockImplementation(async () => {
      setVoicePlaybackState({
        audioElement: null,
        messageId: 'assistant-stream-1',
        sequence: 0,
        source: 'read-aloud',
        status: 'preparing'
      })

      // Holds mid-ladder — the WS-fallback-then-POST round trip the issue
      // describes — until the test rewrites the message id underneath it.
      await new Promise<void>(resolve => {
        settleFallback = resolve
      })

      $messages.set([assistantMessage('durable-42', 'hello there')])

      setVoicePlaybackState({
        audioElement: null,
        messageId: 'durable-42',
        sequence: 0,
        source: 'read-aloud',
        status: 'idle'
      })

      return true
    })

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

    act(() => {
      $messages.set([assistantMessage('assistant-stream-1', 'hello there')])
    })

    await waitFor(() => expect(playSpeechText).toHaveBeenCalledTimes(1))

    await act(async () => {
      settleFallback?.()
      await Promise.resolve()
      await Promise.resolve()
      await Promise.resolve()
    })

    expect($voicePlayback.get().status).toBe('idle')
    expect(playSpeechText).toHaveBeenCalledTimes(1)
  })

  it('speaks a tool-folded rewrite once when the assistant ordinal drifts', async () => {
    $autoSpeakReplies.set(true)

    const $messages = atom<ChatMessage[]>([])

    const pendingReply = () => {
      const messages = $messages.get()
      const last = messages.findLast(m => m.role === 'assistant' && !m.hidden)
      const spoken = resolveSpokenReply(SESSION_ID, messages)

      if (!last || last.id === spoken?.id) {
        return null
      }

      return { id: last.id, pending: Boolean(last.pending), text: chatMessageText(last), turnKey: `${SESSION_ID}:0` }
    }

    const markSpoken = () => {
      const messages = $messages.get()
      const last = messages.findLast(m => m.role === 'assistant' && !m.hidden)

      if (last) {
        markAssistantIdSpoken(SESSION_ID, messages, last.id)
      }
    }

    let settlePlayback: (() => void) | null = null

    vi.mocked(playSpeechText).mockImplementation(async () => {
      setVoicePlaybackState({
        audioElement: null,
        messageId: 'assistant-stream-1',
        sequence: 1,
        source: 'read-aloud',
        status: 'preparing'
      })

      await new Promise<void>(resolve => {
        settlePlayback = resolve
      })

      // Tool rows fold into one durable bubble. The assistant ordinal moves.
      $messages.set([
        { id: 'u1', parts: [assistantTextPart('do the thing')], role: 'user' },
        assistantMessage('durable-42', 'the folded answer')
      ])
      setVoicePlaybackState({ ...IDLE_STATE })

      return true
    })

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

    act(() => {
      $messages.set([
        { id: 'u1', parts: [assistantTextPart('do the thing')], role: 'user' },
        assistantMessage('narration', 'checking'),
        assistantMessage('tool-segment', 'ran the tool'),
        assistantMessage('assistant-stream-1', 'the folded answer')
      ])
    })

    await waitFor(() => expect(playSpeechText).toHaveBeenCalledTimes(1))

    await act(async () => {
      settlePlayback?.()
      await Promise.resolve()
      await Promise.resolve()
      await Promise.resolve()
    })

    expect($voicePlayback.get().status).toBe('idle')
    expect(playSpeechText).toHaveBeenCalledTimes(1)
  })

  it('does not consume a reply when markSpoken runs before a play that never starts', async () => {
    $autoSpeakReplies.set(true)

    const $messages = atom<ChatMessage[]>([])

    const pendingReply = () => {
      const messages = $messages.get()
      const last = messages.findLast(m => m.role === 'assistant' && !m.hidden)
      const spoken = resolveSpokenReply(SESSION_ID, messages)

      if (!last || last.id === spoken?.id) {
        return null
      }

      return { id: last.id, pending: false, text: chatMessageText(last) }
    }

    const markSpoken = () => {
      const messages = $messages.get()
      const last = messages.findLast(m => m.role === 'assistant' && !m.hidden)

      if (last) {
        markAssistantIdSpoken(SESSION_ID, messages, last.id)
      }
    }

    vi.mocked(playSpeechText).mockResolvedValue(false)

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

    act(() => {
      $messages.set([assistantMessage('assistant-stream-1', 'hello there')])
    })

    await waitFor(() => expect(playSpeechText).toHaveBeenCalledTimes(1))
    await waitFor(() => expect(resolveSpokenReply(SESSION_ID, $messages.get())).toBeNull())
  })

  it('speaks a later turn that says the same thing', async () => {
    $autoSpeakReplies.set(true)

    const $messages = atom<ChatMessage[]>([])

    const pendingReply = () => {
      const messages = $messages.get()
      const last = messages.findLast(m => m.role === 'assistant' && !m.hidden)
      const spoken = resolveSpokenReply(SESSION_ID, messages)

      if (!last || last.id === spoken?.id) {
        return null
      }

      return { id: last.id, pending: false, text: chatMessageText(last) }
    }

    const markSpoken = () => {
      const messages = $messages.get()
      const last = messages.findLast(m => m.role === 'assistant' && !m.hidden)

      if (last) {
        markAssistantIdSpoken(SESSION_ID, messages, last.id)
      }
    }

    vi.mocked(playSpeechText).mockResolvedValue(true)

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

    act(() => {
      $messages.set([
        { id: 'u1', parts: [assistantTextPart('again')], role: 'user' },
        assistantMessage('assistant-stream-1', 'Done.')
      ])
    })
    await waitFor(() => expect(playSpeechText).toHaveBeenCalledTimes(1))

    act(() => {
      $messages.set([
        { id: 'u1', parts: [assistantTextPart('again')], role: 'user' },
        assistantMessage('durable-1', 'Done.'),
        { id: 'u2', parts: [assistantTextPart('again')], role: 'user' },
        assistantMessage('assistant-stream-2', 'Done.')
      ])
    })
    await waitFor(() => expect(playSpeechText).toHaveBeenCalledTimes(2))
  })

  // A typed tool turn streams narration bubbles and then the answer; the
  // first narration plays while the rest seal, and hydration folds the turn
  // into one row before playback goes idle. Every part must be heard once.
  it('reads a tool turn in order, each part once, across the fold', async () => {
    $autoSpeakReplies.set(true)

    const $messages = atom<ChatMessage[]>([])
    const claimed = new Set<string>()

    // The main process's cross-window arbiter: a key is owned once.
    window.hermesDesktop = {
      ...window.hermesDesktop,
      claimAmbientCue: async (key: string) => !claimed.has(key) && Boolean(claimed.add(key))
    } as typeof window.hermesDesktop

    // use-composer-voice.ts's auto-speak contract.
    const pendingReply = () => {
      const messages = $messages.get()
      const piece = nextUnspokenTurnPiece(SESSION_ID, messages, chatMessageSpeechText)

      return piece ? { ...piece, turnKey: `${assistantTurnKey(SESSION_ID, messages, piece.id)}@${piece.start}` } : null
    }

    const markSpoken = (reply?: { id: string; through?: string }) => {
      if (reply?.through !== undefined) {
        markTurnSpokenThrough(SESSION_ID, $messages.get(), reply.id, reply.through)
      }
    }

    const settles: (() => void)[] = []

    vi.mocked(playSpeechText).mockImplementation(async () => {
      setVoicePlaybackState({ ...IDLE_STATE, source: 'read-aloud', status: 'speaking' })
      await new Promise<void>(resolve => settles.push(resolve))
      setVoicePlaybackState({ ...IDLE_STATE })

      return true
    })

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

    const asked: ChatMessage = { id: 'u1', parts: [assistantTextPart('what time is it?')], role: 'user' }
    const narration = assistantMessage('assistant-stream-1', 'Let me check the clock.')
    const opened: ChatMessage = { id: 'assistant-stream-2', parts: [], pending: true, role: 'assistant' }

    // The narration seals and the tool's bubble opens in the same breath.
    act(() => {
      $messages.set([asked, narration, opened])
    })

    await waitFor(() => expect(playSpeechText).toHaveBeenCalledTimes(1))

    act(() => {
      $messages.set([
        asked,
        narration,
        assistantMessage('assistant-stream-2', 'Now the host name.'),
        assistantMessage('assistant-stream-3', 'All done.')
      ])
      // Hydration: one durable row for the whole turn, tool calls between the texts.
      $messages.set([
        { ...asked, id: '1-user' },
        {
          id: '2-assistant',
          parts: [
            assistantTextPart('Let me check the clock.'),
            toolCall('date'),
            assistantTextPart('Now the host name.'),
            toolCall('hostname'),
            assistantTextPart('All done.')
          ],
          role: 'assistant'
        }
      ])
    })

    await act(async () => {
      settles.shift()?.()
    })
    await waitFor(() => expect(playSpeechText).toHaveBeenCalledTimes(2))
    await act(async () => {
      settles.shift()?.()
    })
    await act(async () => {
      await Promise.resolve()
    })

    expect(vi.mocked(playSpeechText).mock.calls.map(([text]) => text)).toEqual([
      'Let me check the clock.',
      'Now the host name.\n\nAll done.'
    ])
  })
})
