import { describe, expect, it } from 'vitest'

import {
  assistantTimelineMatch,
  graftRefreshedTailOntoBackfill
} from './transcript-backfill-helpers'
import type { ChatMessage } from '@/lib/chat-messages'

const msg = (over: Partial<ChatMessage> & { id: string; role: ChatMessage['role'] }): ChatMessage =>
  ({
    parts: [],
    timestamp: 1000,
    ...over
  }) as ChatMessage

describe('assistantTimelineMatch temporal fallback', () => {
  it('still matches by identical text first', () => {
    const stored = msg({ id: 'row-1', role: 'assistant', parts: [{ text: 'hello world', type: 'text' }] })
    const local = msg({ id: 'assistant-stream-7', role: 'assistant', parts: [{ text: 'hello world', type: 'text' }] })

    expect(assistantTimelineMatch(stored, local)).toBe(true)
  })

  it('matches by tool-call id when text differs', () => {
    const stored = msg({
      id: 'row-2',
      role: 'assistant',
      parts: [{ text: 'stored tail', type: 'text' }, { toolCallId: 'call_1', type: 'tool-call' } as never]
    })
    const local = msg({
      id: 'assistant-stream-8',
      role: 'assistant',
      parts: [{ toolCallId: 'call_1', type: 'tool-call' } as never]
    })

    expect(assistantTimelineMatch(stored, local)).toBe(true)
  })

  it('falls back to a near timestamp when the streamed text differs from the stored row', () => {
    // The disappearing-agent-message bug: a live stream whose text was
    // truncated/compressed differs from the stored row after normalization,
    // so text matching failed and hydration replaced/dropped the live bubble.
    // The two rows are the same turn when their timestamps agree (the local
    // stream stamps occurredAt = Date.now()/1000; the stored row its ts).
    const stored = msg({
      completedAt: 1_700_000_005,
      id: 'row-3',
      role: 'assistant',
      parts: [{ text: 'full stored reply text', timestamp: 1_700_000_000, type: 'text' }],
      timestamp: 1_700_000_000
    })
    const local = msg({
      completedAt: 1_700_000_006,
      id: 'assistant-stream-9',
      parts: [{ text: 'partial stream…', timestamp: 1_700_000_000.4, type: 'text' }],
      role: 'assistant',
      timestamp: 1_700_000_000.4
    })

    expect(assistantTimelineMatch(stored, local)).toBe(true)
  })

  it('does not match when neither text, tool calls nor time agree', () => {
    const stored = msg({
      id: 'row-4',
      parts: [{ text: 'way earlier turn', timestamp: 1_700_000_000, type: 'text' }],
      role: 'assistant',
      timestamp: 1_700_000_000
    })
    const local = msg({
      id: 'assistant-stream-10',
      parts: [{ text: 'a later turn entirely', timestamp: 1_700_003_600, type: 'text' }],
      role: 'assistant',
      timestamp: 1_700_003_600
    })

    expect(assistantTimelineMatch(stored, local)).toBe(false)
  })
})

describe('graftRefreshedTailOntoBackfill preserved-prefix guard', () => {
  it('keeps a preserved local error pair that trails before the refreshed tail', () => {
    // preserveLocalAssistantErrors APPENDS preserved rows after the merged
    // tail. On the next graft, the previous transcript can therefore END with
    // preserved rows that the refreshed tail legitimately lacks — anchor=0
    // used to return just the tail, silently dropping them ("agent messages
    // disappear").
    const olderStored = msg({ id: 'stored-1', parts: [{ text: 'earlier stored', type: 'text' }], role: 'user' })
    const localError = msg({ error: 'boom', id: 'assistant-stream-3', role: 'assistant' })
    const localUser = msg({ id: 'user-9', parts: [{ text: 'retry?', type: 'text' }], role: 'user' })
    const refreshed = [msg({ id: 'stored-1', parts: [{ text: 'earlier stored', type: 'text' }], role: 'user' })]

    const grafted = graftRefreshedTailOntoBackfill(refreshed, [olderStored, localError, localUser])

    expect(grafted.map(m => m.id)).toEqual(['stored-1', 'assistant-stream-3', 'user-9'])
  })
})
