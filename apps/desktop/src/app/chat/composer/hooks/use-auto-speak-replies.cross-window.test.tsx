// Loaded once at collection, outside every test's budget: vi.resetModules() keeps each module's
// transformed code, so the fresh copy a test imports only re-evaluates it. Without this the first
// test pays the cold transform of the whole graph and times out on a loaded runner.
import '@/lib/chat-messages'
import '@/lib/spoken-reply'
import '@/store/voice-playback'
import '@/store/voice-prefs'
import '../scope'
import './use-auto-speak-replies'

import { act, cleanup, renderHook } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, expect, it, vi } from 'vitest'

import type { ChatMessage } from '@/lib/chat-messages'

import { createAmbientClaimArbiter } from '../../../../../electron/event-dedupe'

// Two windows with the same chat open: each renderer has its own playback
// store, and the only thing they share is the main-process claim arbiter.
// Adapted from @unsupportedpastels' review probe on #140623.

const IDLE = { audioElement: null, messageId: null, sequence: 0, source: null, status: 'idle' as const }

const message = (id: string, text: string, role: 'assistant' | 'user' = 'assistant'): ChatMessage => ({
  id,
  parts: [{ text, type: 'text' }],
  role
})

afterEach(() => {
  cleanup()
  delete (window as { hermesDesktop?: unknown }).hermesDesktop
  vi.doUnmock('@/lib/voice-playback')
  vi.resetModules()
})

async function openWindow(name: string, plays: string[]) {
  vi.resetModules()
  const playback = await import('@/store/voice-playback')
  const prefs = await import('@/store/voice-prefs')
  const spoken = await import('@/lib/spoken-reply')
  const chat = await import('@/lib/chat-messages')
  const scope = await import('../scope')

  vi.doMock('@/lib/voice-playback', () => ({
    playSpeechText: vi.fn(async (text: string) => {
      plays.push(`${name}:${text}`)
      playback.setVoicePlaybackState({ ...IDLE, source: 'read-aloud', status: 'speaking' })

      return true
    })
  }))

  const { useAutoSpeakReplies } = await import('./use-auto-speak-replies')
  const user = message('u1', 'fix it', 'user')
  const $messages = atom<ChatMessage[]>([user])
  prefs.$autoSpeakReplies.set(true)

  const markSpoken = () => {
    const rows = $messages.get()
    const last = rows.findLast(m => m.role === 'assistant')

    if (last) {
      spoken.markAssistantIdSpoken('s1', rows, last.id)
    }
  }

  const pendingReply = () => {
    const rows = $messages.get()
    const last = rows.findLast(m => m.role === 'assistant')

    if (!last || last.id === spoken.resolveSpokenReply('s1', rows)?.id) {
      return null
    }

    return {
      id: last.id,
      pending: false,
      segmentText: chat.lastTextPartText(last),
      text: chat.chatMessageText(last),
      turnKey: spoken.assistantTurnKey('s1', rows, last.id)
    }
  }

  renderHook(
    () =>
      useAutoSpeakReplies({
        conversationActive: false,
        failureLabel: 'failed',
        markSpoken,
        pendingReply,
        sessionId: 's1'
      }),
    {
      wrapper: ({ children }) => (
        <scope.ComposerScopeProvider value={{ ...scope.MAIN_COMPOSER_SCOPE, $messages }}>
          {children}
        </scope.ComposerScopeProvider>
      )
    }
  )

  return { $messages, playback, user }
}

it('reads the final reply once, in the window that read the progress update, after that clip ends', async () => {
  const owns = createAmbientClaimArbiter()

  ;(window as { hermesDesktop?: unknown }).hermesDesktop = { claimAmbientCue: async (key: string) => owns(key) }

  const plays: string[] = []
  const a = await openWindow('A', plays)
  const b = await openWindow('B', plays)
  expect(a.playback.$voicePlayback).not.toBe(b.playback.$voicePlayback)

  const progress = message('assistant-interim-1', 'Looking into it.')
  const final = message('assistant-stream-2', 'Here is the final report.')

  await act(async () => {
    a.$messages.set([a.user, progress])
  })
  await act(async () => {
    b.$messages.set([b.user, progress])
  })
  expect(plays).toEqual(['A:Looking into it.'])

  // The final reply lands in both windows while A is still speaking.
  await act(async () => {
    a.$messages.set([a.user, progress, final])
    b.$messages.set([b.user, progress, final])
  })
  expect(plays).toEqual(['A:Looking into it.'])

  // A's progress clip ends.
  await act(async () => {
    a.playback.setVoicePlaybackState({ ...IDLE })
  })
  expect(plays).toEqual(['A:Looking into it.', 'A:Here is the final report.'])
})
