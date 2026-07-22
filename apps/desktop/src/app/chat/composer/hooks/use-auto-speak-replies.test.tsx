import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { playSpeechText } from '@/lib/voice-playback'
import { $messages } from '@/store/session'
import { setVoicePlaybackState } from '@/store/voice-playback'
import { $autoSpeakReplies } from '@/store/voice-prefs'

import { useAutoSpeakReplies } from './use-auto-speak-replies'

vi.mock('@/lib/voice-playback', () => ({ playSpeechText: vi.fn(() => Promise.resolve(true)) }))
vi.mock('@/store/notifications', () => ({ notifyError: vi.fn() }))

interface Reply {
  id: string
  pending: boolean
  text: string
}

describe('useAutoSpeakReplies', () => {
  afterEach(() => {
    cleanup()
    $autoSpeakReplies.set(false)
    $messages.set([])
    setVoicePlaybackState({
      audioElement: null,
      messageId: null,
      sequence: 0,
      source: null,
      status: 'idle'
    })
    vi.clearAllMocks()
  })

  it('does not replay history that hydrates after selecting a session', () => {
    let reply: Reply | null = { id: 'assistant-a', pending: false, text: 'Already heard A' }
    let spokenId: string | null = null

    const markSpoken = () => {
      spokenId = reply?.id ?? null
    }

    const pendingReply = () => (reply?.id === spokenId ? null : reply)

    $autoSpeakReplies.set(true)

    const { rerender } = renderHook(
      ({ sessionId }) =>
        useAutoSpeakReplies({
          busy: false,
          conversationActive: false,
          failureLabel: 'Read aloud failed',
          markSpoken,
          pendingReply,
          sessionId
        }),
      { initialProps: { sessionId: 'session-a' } }
    )

    rerender({ sessionId: 'session-b' })

    act(() => {
      reply = { id: 'assistant-b-history', pending: false, text: 'Existing session B reply' }
      $messages.set([])
    })

    expect(playSpeechText).not.toHaveBeenCalled()

    act(() => {
      reply = { id: 'assistant-b-new', pending: true, text: 'A new reply' }
      $messages.set([])
    })
    expect(playSpeechText).not.toHaveBeenCalled()

    act(() => {
      reply = { id: 'assistant-b-new', pending: false, text: 'A new reply' }
      $messages.set([])
    })

    expect(playSpeechText).toHaveBeenCalledOnce()
    expect(playSpeechText).toHaveBeenCalledWith('A new reply', {
      messageId: 'assistant-b-new',
      source: 'read-aloud'
    })
  })

  it('speaks an atomic completed reply after a new user turn', () => {
    let reply: Reply | null = { id: 'assistant-history', pending: false, text: 'Existing reply' }
    let spokenId: string | null = null

    const markSpoken = () => {
      spokenId = reply?.id ?? null
    }

    const pendingReply = () => (reply?.id === spokenId ? null : reply)

    $autoSpeakReplies.set(true)

    const { rerender } = renderHook(
      ({ busy }) =>
        useAutoSpeakReplies({
          busy,
          conversationActive: false,
          failureLabel: 'Read aloud failed',
          markSpoken,
          pendingReply,
          sessionId: 'session-a'
        }),
      { initialProps: { busy: false } }
    )

    rerender({ busy: true })

    act(() => {
      reply = { id: 'assistant-new', pending: false, text: 'Atomic reply' }
      $messages.set([{ id: 'assistant-new', role: 'assistant', content: 'Atomic reply' }] as never)
    })

    expect(playSpeechText).toHaveBeenCalledWith('Atomic reply', {
      messageId: 'assistant-new',
      source: 'read-aloud'
    })
  })
})
