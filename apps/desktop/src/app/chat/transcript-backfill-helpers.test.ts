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

  it('falls back to a near timestamp when the streamed text differs AND a side is textless', () => {
    // The disappearing-agent-message bug: a live stream that produced no
    // comparable text (tool-only turn, truncated fragment) differs from the
    // stored row after normalization, so text matching failed and hydration
    // replaced the live bubble. The two rows are the same turn when their
    // timestamps agree (the local stream stamps occurredAt = Date.now()/1000;
    // the stored row its ts) and at least one side carries no real text.
    const stored = msg({
      completedAt: 1_700_000_005,
      id: 'row-3',
      parts: [{ text: 'full stored reply text', timestamp: 1_700_000_000, type: 'text' }],
      role: 'assistant',
      timestamp: 1_700_000_000
    })
    const local = msg({
      completedAt: 1_700_000_006,
      id: 'assistant-stream-9',
      parts: [],
      role: 'assistant',
      timestamp: 1_700_000_000.4
    })

    expect(assistantTimelineMatch(stored, local)).toBe(true)
  })

  it('falls back to a near timestamp only when a side is textless (no false match between distinct text turns)', () => {
    // Two DISTINCT turns within the 2s window, both with real text: time alone
    // must not match them (the old temporal rule could), or hydration would
    // reconcile the wrong rows.
    const stored = msg({
      id: 'row-3a',
      parts: [{ text: 'first answer', timestamp: 1_700_000_000, type: 'text' }],
      role: 'assistant',
      timestamp: 1_700_000_000
    })
    const local = msg({
      id: 'assistant-stream-11',
      parts: [{ text: 'second answer', timestamp: 1_700_000_001.5, type: 'text' }],
      role: 'assistant',
      timestamp: 1_700_000_001.5
    })

    expect(assistantTimelineMatch(stored, local)).toBe(false)
  })

  it('still temporal-matches a textless live fragment to its stored row', () => {
    // A live stream whose only part is a tool call or an empty fragment: text
    // matching had nothing to compare, so the near timestamp carries it.
    const stored = msg({
      id: 'row-3b',
      parts: [{ text: 'full stored reply text', timestamp: 1_700_000_000, type: 'text' }],
      role: 'assistant',
      timestamp: 1_700_000_000
    })
    const local = msg({
      id: 'assistant-stream-12',
      parts: [],
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

  it('keeps a settled reply the refreshed page no longer carries (anchorless)', () => {
    // Compaction rewrite: every row id in the transcript is new, so the anchor
    // lookup cannot match anything and the refreshed tail is adopted outright.
    // A read landing while the rewrite is in flight (or lagging it) returns a
    // page without the newest settled reply — adopting the page wholesale
    // drops a reply the user already saw (same "vanishing message" family as
    // the anchor === 0 defect, different branch).
    const previous = [
      msg({ id: 'u1', parts: [{ text: 'the prompt', type: 'text' }], role: 'user', timestamp: 1_900 }),
      msg({ id: 'a1', parts: [{ text: 'the answer', type: 'text' }], role: 'assistant', timestamp: 2_000 })
    ]
    const refreshed = [
      msg({ id: 'p', parts: [{ text: 'an older turn', type: 'text' }], role: 'assistant', timestamp: 1_000 }),
      msg({ id: 'q', parts: [{ text: 'an older reply', type: 'text' }], role: 'assistant', timestamp: 1_400 })
    ]

    expect(graftRefreshedTailOntoBackfill(refreshed, previous).map(m => m.id)).toEqual(['p', 'q', 'a1'])
  })

  it('does not re-append a settled reply the page has genuinely moved past', () => {
    // The page carries a strictly newer row AND already contains the reply's
    // text — the page wins, nothing is duplicated.
    const previous = [
      msg({ id: 'u1', parts: [{ text: 'the prompt', type: 'text' }], role: 'user', timestamp: 1_900 }),
      msg({ id: 'a1', parts: [{ text: 'the answer', type: 'text' }], role: 'assistant', timestamp: 2_000 })
    ]
    const refreshed = [
      msg({ id: 'q', parts: [{ text: 'an older reply', type: 'text' }], role: 'assistant', timestamp: 1_400 }),
      msg({ id: 'r', parts: [{ text: 'the answer', type: 'text' }], role: 'assistant', timestamp: 2_100 })
    ]

    expect(graftRefreshedTailOntoBackfill(refreshed, previous).map(m => m.id)).toEqual(['q', 'r'])
  })

  it('keeps reference identity when there is no settled reply to hold back', () => {
    const previous = [msg({ id: 'u1', parts: [{ text: 'the prompt', type: 'text' }], role: 'user', timestamp: 1_900 })]
    const refreshed = [
      msg({ id: 'p', parts: [{ text: 'an older turn', type: 'text' }], role: 'assistant', timestamp: 1_000 })
    ]

    expect(graftRefreshedTailOntoBackfill(refreshed, previous)).toBe(refreshed)
  })
})
