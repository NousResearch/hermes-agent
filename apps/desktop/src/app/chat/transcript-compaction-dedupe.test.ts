import { renderHook } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { graftRefreshedTailOntoBackfill } from '@/app/chat/transcript-backfill'
import type { ChatMessage } from '@/lib/chat-messages'

import { useRuntimeMessageRepository } from './runtime-repository'

/**
 * The shapes a compaction generation pair takes on the way to the screen,
 * measured against a live store in #123985. The refresh read
 * (`include_compacted=true`) ships both generations, and
 * `mergeOverlappingTail` unions the window with the refreshed page by durable
 * row id, so whether the pair survives the merge depends on whether the window
 * and the page share one post-compaction row id — not on the row shape itself.
 *
 * Case 1 is the one neither client PR covers: the merge keeps both generations,
 * and only a content-level key at the repository boundary collapses them.
 * Cases 3 and 4 are guards on that key.
 */

const TIMESTAMP = 1_771_500_000

/** A hydrated chat row: distinct string ids, durable `rowId` per backend row. */
const chat = (
  id: string,
  rowId: number,
  role: 'user' | 'assistant',
  text: string,
  timestamp: number | null = TIMESTAMP
): ChatMessage => ({
  id,
  role,
  parts: [{ type: 'text', text }],
  rowId,
  ...(timestamp === null ? {} : { timestamp })
})

/** The same row without a durable id (a live/streaming copy). */
const live = (
  id: string,
  role: 'user' | 'assistant',
  text: string,
  timestamp: number | null = TIMESTAMP
): ChatMessage => ({
  id,
  role,
  parts: [{ type: 'text', text }],
  ...(timestamp === null ? {} : { timestamp })
})

const rendered = (messages: ChatMessage[]): string[] =>
  messages.flatMap(message => message.parts.flatMap(part => (part.type === 'text' ? [part.text] : [])))

interface RuntimePart {
  type: string
  text?: string
}

const render = (messages: ChatMessage[]): string[] =>
  renderHook(() => useRuntimeMessageRepository(messages)).result.current.messages.flatMap(item =>
    (item.message.content as readonly RuntimePart[]).flatMap(part =>
      part.type === 'text' && part.text !== undefined ? [part.text] : []
    )
  )

describe('compaction generations reaching the transcript', () => {
  it('renders one copy when the window shares a post-compaction row id with the refreshed page', () => {
    // The pre-compaction generation (row ids 100/101) is still in the live
    // window; the row persisted after the compaction (202) is on both sides, so
    // the merge unions by row id and both generations travel to the repository.
    const window = [
      chat('w-100', 100, 'user', 'Q'),
      chat('w-101', 101, 'assistant', 'A'),
      chat('w-202', 202, 'assistant', 'AFTER')
    ]
    const page = [
      chat('p-200', 200, 'user', 'Q'),
      chat('p-201', 201, 'assistant', 'A'),
      chat('p-202', 202, 'assistant', 'AFTER')
    ]

    const merged = graftRefreshedTailOntoBackfill(page, window)

    // The merge itself keeps the pair — that is what the repository key is for.
    expect(rendered(merged)).toEqual(['Q', 'A', 'Q', 'A', 'AFTER'])
    expect(render(merged)).toEqual(['Q', 'A', 'AFTER'])
  })

  it('renders one copy when the page shares no durable row id with the window', () => {
    // Same logical transcript, every id rebuilt and no post-compaction row in
    // the window: the page replaces the window, so nothing is left to dedupe.
    const window = [chat('w-100', 100, 'user', 'Q'), chat('w-101', 101, 'assistant', 'A')]
    const page = [chat('p-200', 200, 'user', 'Q'), chat('p-201', 201, 'assistant', 'A')]

    const merged = graftRefreshedTailOntoBackfill(page, window)

    expect(rendered(merged)).toEqual(['Q', 'A'])
    expect(render(merged)).toEqual(['Q', 'A'])
  })

  it('never collapses two rows on an absent timestamp', () => {
    // The trap: a key that substitutes a placeholder for a missing timestamp
    // merges two unrelated rows here.
    const messages = [live('a-1', 'assistant', 'same words'), live('a-2', 'assistant', 'same words', null)]

    expect(render(messages)).toEqual(['same words', 'same words'])
  })

  it('keeps two identical replies at different timestamps apart', () => {
    // Measured shape: many groups share role/content and differ only in the
    // timestamp, so the key has to keep the timestamp.
    const messages = [
      chat('b-1', 1, 'assistant', 'same words', TIMESTAMP),
      chat('b-2', 2, 'assistant', 'same words', TIMESTAMP + 3_600)
    ]

    expect(render(messages)).toEqual(['same words', 'same words'])
  })
})
