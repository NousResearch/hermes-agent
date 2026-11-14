// Turning read-aloud off must silence the reply it is reading right now, in this
// window and when another window flips the switch (Discord report: "the off
// toggle does nothing"). A reply started by hand from its message button is not
// the toggle's to stop.

import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { assistantTextPart, type ChatMessage } from '@/lib/chat-messages'
import { clearSpokenRepliesForTests } from '@/lib/spoken-reply'
import { playSpeechText, stopVoicePlayback } from '@/lib/voice-playback'
import { setVoicePlaybackState } from '@/store/voice-playback'
import { $autoSpeakReplies } from '@/store/voice-prefs'

import { ComposerScopeProvider, MAIN_COMPOSER_SCOPE } from '../scope'

import { clearOwnedSpeechTurnsForTests, useAutoSpeakReplies } from './use-auto-speak-replies'

vi.mock('@/lib/voice-playback', () => ({
  playSpeechText: vi.fn(),
  stopVoicePlayback: vi.fn()
}))

const IDLE = { audioElement: null, messageId: null, sequence: 0, source: null, status: 'idle' as const }

function reply(id: string): ChatMessage {
  return { id, parts: [assistantTextPart('a long answer being read')], role: 'assistant' }
}

function mountReading() {
  const $messages = atom<ChatMessage[]>([])
  let spoken: string | null = null

  vi.mocked(playSpeechText).mockImplementation(async (_text, options) => {
    setVoicePlaybackState({ ...IDLE, messageId: options?.messageId ?? null, source: 'read-aloud', status: 'speaking' })

    return true
  })

  renderHook(
    () =>
      useAutoSpeakReplies({
        conversationActive: false,
        failureLabel: 'read-aloud failed',
        markSpoken: () => {
          spoken = $messages.get().at(-1)?.id ?? null
        },
        pendingReply: () => {
          const last = $messages.get().at(-1)

          return last && last.id !== spoken ? { id: last.id, pending: false, text: 'a long answer being read' } : null
        },
        sessionId: 'session-off'
      }),
    {
      wrapper: ({ children }) => (
        <ComposerScopeProvider value={{ ...MAIN_COMPOSER_SCOPE, $messages }}>{children}</ComposerScopeProvider>
      )
    }
  )

  return $messages
}

describe('turning read-aloud off stops the reply being read', () => {
  afterEach(() => {
    cleanup()
    clearSpokenRepliesForTests()
    clearOwnedSpeechTurnsForTests()
    $autoSpeakReplies.set(false)
    setVoicePlaybackState({ ...IDLE })
    vi.clearAllMocks()
  })

  it.each([
    ['in this window', () => $autoSpeakReplies.set(false)],
    [
      'from another window',
      () => {
        localStorage.setItem('hermes.desktop.autoSpeakReplies', 'false')
        window.dispatchEvent(
          new StorageEvent('storage', { key: 'hermes.desktop.autoSpeakReplies', newValue: 'false', oldValue: 'true' })
        )
      }
    ]
  ])('%s', async (_where, turnOff) => {
    $autoSpeakReplies.set(true)
    const $messages = mountReading()

    act(() => $messages.set([reply('r1')]))
    await waitFor(() => expect(playSpeechText).toHaveBeenCalledTimes(1))

    act(turnOff)

    expect($autoSpeakReplies.get()).toBe(false)
    expect(stopVoicePlayback).toHaveBeenCalledTimes(1)
  })

  it('leaves a reply the user started from its own button playing', async () => {
    $autoSpeakReplies.set(true)
    mountReading()

    // The message's own "read aloud" button, not the auto-speak hook.
    setVoicePlaybackState({ ...IDLE, messageId: 'older', source: 'read-aloud', status: 'speaking' })

    act(() => $autoSpeakReplies.set(false))

    expect(stopVoicePlayback).not.toHaveBeenCalled()
  })
})
