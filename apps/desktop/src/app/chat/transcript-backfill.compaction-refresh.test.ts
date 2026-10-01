import { describe, expect, it } from 'vitest'

import { graftRefreshedTailOntoBackfill } from '@/app/chat/transcript-backfill'
import { preserveLocalPendingTurnMessages } from '@/app/session/hooks/use-session-actions/utils'
import { type ChatMessage, chatMessageText, toChatMessages } from '@/lib/chat-messages'
import type { SessionMessage } from '@/types/hermes'

/**
 * Background refresh / post-turn rehydrate over a retention-trimmed store.
 *
 * `transcript-retention` releases the store's head once it is off screen, so
 * in a long session the newest REST page (120 display rows) begins at a row the
 * store no longer holds. The graft then has no anchor and falls through to the
 * stored-id merge. The payloads below are the real display projection
 * (`SessionDB.get_messages(include_compacted=True, latest=True)`) around an
 * in-place compaction: the carried tail is re-inserted under fresh ids and the
 * originals are superseded out of the projection.
 */

const T0 = 1_790_000_000

const stored = (id: number, role: 'assistant' | 'user', content: string, timestamp: number): SessionMessage => ({
  content,
  id,
  role,
  timestamp
})

// Before compaction: q1..a4 as rows 1..8.
const beforeCompaction: SessionMessage[] = [
  stored(1, 'user', 'question 1', T0 + 10),
  stored(2, 'assistant', 'answer 1', T0 + 11),
  stored(3, 'user', 'question 2', T0 + 20),
  stored(4, 'assistant', 'answer 2', T0 + 21),
  stored(5, 'user', 'question 3', T0 + 30),
  stored(6, 'assistant', 'answer 3', T0 + 31),
  stored(7, 'user', 'question 4', T0 + 40),
  stored(8, 'assistant', 'answer 4', T0 + 41)
]

// After `archive_and_compact(summary + carried q4/a4)` and one more turn: the
// carried q4/a4 are rows 10/11 now, 7/8 left the projection.
const afterCompaction: SessionMessage[] = [
  ...beforeCompaction.slice(0, 6),
  {
    ...stored(9, 'user', '[CONTEXT COMPACTION — REFERENCE ONLY] summary of earlier turns', T0 + 100),
    display_kind: 'hidden'
  },
  stored(10, 'user', 'question 4', T0 + 40),
  stored(11, 'assistant', 'answer 4', T0 + 41),
  stored(12, 'user', 'question 5', T0 + 200),
  stored(13, 'assistant', 'answer 5', T0 + 201)
]

const texts = (messages: ChatMessage[]) => messages.map(message => `${message.role}:${chatMessageText(message)}`)

describe('refresh merge over a retention-trimmed store', () => {
  it('does not keep a compaction-superseded row next to its re-inserted copy (#122167, #126229)', () => {
    // Retention released q1/a1: the store starts at row 3.
    const retained = toChatMessages(beforeCompaction).slice(2)
    const refreshed = toChatMessages(afterCompaction)

    expect(texts(graftRefreshedTailOntoBackfill(refreshed, retained))).toEqual([
      'user:question 1',
      'assistant:answer 1',
      'user:question 2',
      'assistant:answer 2',
      'user:question 3',
      'assistant:answer 3',
      'user:question 4',
      'assistant:answer 4',
      'user:question 5',
      'assistant:answer 5'
    ])
  })

  it('lets the committed turn replace its settled live rows instead of pinning them below it (#126229)', () => {
    const retained = toChatMessages(beforeCompaction).slice(2)

    // The turn this window just streamed: an optimistic prompt and a settled
    // stream bubble, neither carrying a stored id yet.
    const window: ChatMessage[] = [
      ...retained,
      { id: 'user-1790000200000-abc', parts: [{ text: 'question 5', type: 'text' }], role: 'user' },
      {
        id: 'assistant-stream-1790000200500-1',
        parts: [{ text: 'answer 5', type: 'text' }],
        pending: false,
        role: 'assistant'
      }
    ]

    const refreshed = toChatMessages([
      ...beforeCompaction,
      stored(9, 'user', 'question 5', T0 + 200),
      stored(10, 'assistant', 'answer 5', T0 + 201)
    ])

    const merged = preserveLocalPendingTurnMessages(graftRefreshedTailOntoBackfill(refreshed, window), window)

    expect(texts(merged)).toEqual([
      'user:question 1',
      'assistant:answer 1',
      'user:question 2',
      'assistant:answer 2',
      'user:question 3',
      'assistant:answer 3',
      'user:question 4',
      'assistant:answer 4',
      'user:question 5',
      'assistant:answer 5'
    ])
  })
})
