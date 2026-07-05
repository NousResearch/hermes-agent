import { act, renderHook } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, describe, expect, it, vi } from 'vitest'

const stopVoicePlayback = vi.fn()
const playSpeechText = vi.fn().mockResolvedValue(true)
const notifyError = vi.fn()

vi.mock('@/lib/voice-playback', () => ({
  playSpeechText: (...args: unknown[]) => playSpeechText(...args),
  stopVoicePlayback: (...args: unknown[]) => stopVoicePlayback(...args)
}))
vi.mock('@/store/notifications', () => ({ notifyError: (...args: unknown[]) => notifyError(...args) }))

// Mutable stores — tests swap .set() to simulate session switches.
const $messages = atom<Array<{ id: string; role: string; hidden?: boolean }>>([])
const $voicePlayback = atom<{ status: string }>({ status: 'idle' })
const $autoSpeakReplies = atom(false)

vi.mock('@/store/session', () => ({ $messages }))
vi.mock('@/store/voice-playback', () => ({ $voicePlayback }))
vi.mock('@/store/voice-prefs', () => ({ $autoSpeakReplies }))

import { useAutoSpeakReplies } from './use-auto-speak-replies'

afterEach(() => {
  vi.clearAllMocks()
})

function renderAutoSpeak(overrides: Partial<Parameters<typeof useAutoSpeakReplies>[0]> = {}) {
  const markSpoken = vi.fn()
  let currentSessionMessages = $messages.get()
  const pendingReply = () => {
    const msgs = $messages.get()
    const last = msgs.findLast(m => m.role === 'assistant' && !m.hidden)
    if (!last) return null
    return { id: last.id, pending: false, text: `text-${last.id}` }
  }

  const props = {
    conversationActive: false,
    failureLabel: 'TTS failed',
    markSpoken,
    pendingReply,
    sessionId: 'session-A' as string | null | undefined,
    ...overrides
  }

  return renderHook(
    ({ sessionId }) => useAutoSpeakReplies({ ...props, sessionId }),
    { initialProps: { sessionId: props.sessionId } }
  )
}

describe('useAutoSpeakReplies', () => {
  it('calls stopVoicePlayback on effect cleanup when sessionId changes', () => {
    $autoSpeakReplies.set(true)
    $messages.set([{ id: 'a1', role: 'assistant' }])

    const { rerender } = renderAutoSpeak()

    // Switch session — triggers old effect cleanup + new effect setup
    act(() => {
      $messages.set([{ id: 'b1', role: 'assistant' }])
      rerender({ sessionId: 'session-B' })
    })

    // stopVoicePlayback must be called during cleanup to silence any playback
    // that slipped through from the stale subscription.
    expect(stopVoicePlayback).toHaveBeenCalled()
  })

  it('stops stale playback when a session switch triggers speakLatest before cleanup', () => {
    $autoSpeakReplies.set(true)

    // Start with session A having a reply
    $messages.set([{ id: 'a1', role: 'assistant' }])
    const { rerender } = renderAutoSpeak()

    // Simulate the race: $messages is updated synchronously (by the session
    // switch handler) BEFORE React runs effect cleanup.  The stale
    // subscription fires, starts playback, then cleanup stops it.
    act(() => {
      // This triggers $messages subscribers synchronously — the stale
      // speakLatest sees session B's reply and calls playSpeechText.
      $messages.set([{ id: 'b1', role: 'assistant' }])

      // React re-renders with new sessionId → old effect cleanup runs
      // → stopVoicePlayback() silences the stale playback.
      rerender({ sessionId: 'session-B' })
    })

    expect(stopVoicePlayback).toHaveBeenCalled()
  })

  it('does not call stopVoicePlayback when enabled is false', () => {
    $autoSpeakReplies.set(false)
    $messages.set([{ id: 'a1', role: 'assistant' }])

    const { rerender } = renderAutoSpeak()

    act(() => {
      $messages.set([{ id: 'b1', role: 'assistant' }])
      rerender({ sessionId: 'session-B' })
    })

    // Effect returns early when disabled — no cleanup to run
    expect(stopVoicePlayback).not.toHaveBeenCalled()
  })
})
