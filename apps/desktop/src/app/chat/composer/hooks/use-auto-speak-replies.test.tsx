import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { assistantTextPart, type ChatMessage, chatMessageText } from '@/lib/chat-messages'
import { clearSpokenRepliesForTests, markAssistantIdSpoken, resolveSpokenReply } from '@/lib/spoken-reply'
import { playSpeechText, type SpeechStreamSession, startSpeechStream, stopVoicePlayback } from '@/lib/voice-playback'
import { $voicePlayback, setVoicePlaybackState } from '@/store/voice-playback'
import { $autoSpeakReplies } from '@/store/voice-prefs'

import { ComposerScopeProvider, MAIN_COMPOSER_SCOPE } from '../scope'

import { useAutoSpeakReplies } from './use-auto-speak-replies'

vi.mock('@/lib/voice-playback', () => ({
  playSpeechText: vi.fn(),
  startSpeechStream: vi.fn(async () => null),
  stopVoicePlayback: vi.fn(() => {
    setVoicePlaybackState({ ...IDLE_STATE, sequence: $voicePlayback.get().sequence + 1 })
  })
}))

const SESSION_ID = 'session-under-test'
const IDLE_STATE = { audioElement: null, messageId: null, sequence: 0, source: null, status: 'idle' as const }

function assistantMessage(id: string, text: string): ChatMessage {
  return { id, parts: [assistantTextPart(text)], role: 'assistant' }
}

function renderAutoSpeech($messages = atom<ChatMessage[]>([])) {
  $autoSpeakReplies.set(true)

  const pendingReply = () => {
    const messages = $messages.get()
    const last = messages.findLast(m => m.role === 'assistant' && !m.hidden)

    if (!last || resolveSpokenReply(SESSION_ID, messages)?.id === last.id) {
      return null
    }

    return { id: last.id, pending: Boolean(last.pending), text: chatMessageText(last) }
  }

  const markSpoken = () => {
    const last = $messages.get().findLast(m => m.role === 'assistant' && !m.hidden)

    if (last) {
      markAssistantIdSpoken(SESSION_ID, $messages.get(), last.id)
    }
  }

  const hook = renderHook(
    () =>
      useAutoSpeakReplies({
        conversationActive: false,
        failureLabel: 'failed',
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

  return { $messages, hook }
}

function mockStream() {
  let finishAudio: (value: 'done' | 'fallback') => void = () => undefined

  const session: SpeechStreamSession = {
    append: vi.fn(),
    finish: vi.fn(),
    cancel: vi.fn(() => finishAudio('done')),
    done: new Promise(resolve => {
      finishAudio = resolve
    })
  }

  vi.mocked(startSpeechStream).mockImplementationOnce(async () => {
    setVoicePlaybackState({
      ...IDLE_STATE,
      sequence: $voicePlayback.get().sequence + 1,
      source: 'read-aloud',
      status: 'preparing'
    })

    return session
  })

  return { session, finishAudio }
}

describe('auto-speak streams live replies and respects interruption', () => {
  afterEach(() => {
    cleanup()
    clearSpokenRepliesForTests()
    $autoSpeakReplies.set(false)
    setVoicePlaybackState({ ...IDLE_STATE })
    vi.clearAllMocks()
  })

  it('sends the first sentence before completion and only appends new text after an id rewrite', async () => {
    const { session, finishAudio } = mockStream()
    const { $messages } = renderAutoSpeech()
    act(() => {
      $messages.set([{ ...assistantMessage('assistant-stream-1', 'うん。'), pending: true }])
    })
    await waitFor(() => expect(session.append).toHaveBeenCalledWith('うん。'))
    expect(session.finish).not.toHaveBeenCalled()

    act(() => {
      $messages.set([assistantMessage('durable-1', 'うん。次の文です。')])
    })
    expect(session.append).toHaveBeenLastCalledWith('次の文です。')
    expect(session.finish).toHaveBeenCalledTimes(1)
    await act(async () => {
      finishAudio('done')
    })
    act(() => {
      setVoicePlaybackState({ ...$voicePlayback.get(), status: 'idle' })
    })
    expect(startSpeechStream).toHaveBeenCalledTimes(1)
    expect(playSpeechText).not.toHaveBeenCalled()
  })

  it('keeps streaming the same reply when older history is prepended', async () => {
    const { session } = mockStream()
    const user: ChatMessage = { id: 'current-user', role: 'user', parts: [] }
    const { $messages } = renderAutoSpeech(atom<ChatMessage[]>([user]))
    act(() => {
      $messages.set([user, { ...assistantMessage('live', 'うん。'), pending: true }])
    })
    await waitFor(() => expect(session.append).toHaveBeenCalledWith('うん。'))
    act(() => {
      $messages.set([
        { id: 'older-user', role: 'user', parts: [] },
        assistantMessage('older-assistant', '前の会話です。'),
        user,
        assistantMessage('durable', 'うん。続きです。')
      ])
    })
    expect(session.append).toHaveBeenLastCalledWith('続きです。')
    expect(session.cancel).not.toHaveBeenCalled()
    expect(startSpeechStream).toHaveBeenCalledTimes(1)
  })

  it('does not restart or fall back after Stop while the rest of the reply arrives', async () => {
    const { session } = mockStream()
    const { $messages } = renderAutoSpeech()
    act(() => {
      $messages.set([{ ...assistantMessage('assistant-stream-1', 'うん。'), pending: true }])
    })
    await waitFor(() => expect(session.append).toHaveBeenCalledWith('うん。'))
    act(() => {
      stopVoicePlayback()
    })
    act(() => {
      $messages.set([assistantMessage('durable-1', 'うん。まだ続けます。')])
    })
    await act(async () => {
      await Promise.resolve()
    })
    expect(session.cancel).toHaveBeenCalledTimes(1)
    expect(startSpeechStream).toHaveBeenCalledTimes(1)
    expect(playSpeechText).not.toHaveBeenCalled()
  })

  it('cancels the old reply on new input and lets the next turn speak', async () => {
    const first = mockStream()
    const { $messages } = renderAutoSpeech()
    act(() => {
      $messages.set([assistantMessage('assistant-stream-1', '一文目です。')])
    })
    await waitFor(() => expect(first.session.append).toHaveBeenCalled())
    const user: ChatMessage = { id: 'user-2', role: 'user', parts: [] }
    act(() => {
      $messages.set([...$messages.get(), user])
    })
    expect(first.session.cancel).toHaveBeenCalledTimes(1)
    const second = mockStream()
    act(() => {
      $messages.set([
        ...$messages.get(),
        { ...assistantMessage('assistant-stream-2', '次の回答です。'), pending: true }
      ])
    })
    await waitFor(() => expect(second.session.append).toHaveBeenCalledWith('次の回答です。'))
    expect(startSpeechStream).toHaveBeenCalledTimes(2)
    expect(second.session.cancel).not.toHaveBeenCalled()
  })
})

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
})
