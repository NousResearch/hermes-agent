import { describe, expect, it } from 'vitest'

import { type ChatMessage, toChatMessages } from '@/lib/chat-messages'
import type { SessionMessage } from '@/types/hermes'

// Synthetic display pages of one scripted session, in the shape the REST
// history endpoint returns them: before an in-place compaction, after it, and
// after one more turn. 12 turns; the compaction summarizes the head and
// carries the last 4 rows (prompt/reply 11 and 12) as fresh rows 26-29 with
// the originals' uid, content and timestamp.
import pages from './fixtures/compaction-pages.json'
import { graftRefreshedTailOntoBackfill, mergeOlderTranscriptPage, unifyCompactionCopies } from './transcript-backfill'

interface Page {
  messages: unknown[]
}

const rows = (page: Page) => page.messages as SessionMessage[]
const hydrate = (page: Page): ChatMessage[] => toChatMessages(rows(page))

// The gateway's projection of the same rows (session.resume, session.compress):
// text, timestamp and row_id, but no message_uid.
const gatewayHydrate = (page: Page): ChatMessage[] =>
  toChatMessages(
    rows(page)
      .filter(row => row.display_kind !== 'hidden')
      .map((row): SessionMessage => ({
        content: null,
        role: row.role,
        row_id: row.id,
        text: row.content,
        timestamp: row.timestamp
      }))
  )

const texts = (messages: ChatMessage[]) =>
  messages.map(message => message.parts.map(part => (part.type === 'text' ? part.text : '')).join(''))

/** Every logical message (fixture texts are unique) must have exactly one store entry. */
function expectOneEntryPerMessage(messages: ChatMessage[], expected: Page) {
  const seen = texts(messages)
  const repeated = seen.filter((text, index) => seen.indexOf(text) !== index)

  expect(repeated).toEqual([])
  expect(seen).toEqual(texts(hydrate(expected)).filter(text => seen.includes(text)))
}

describe('compaction copies keep one store entry per message', () => {
  const before = hydrate(pages.before.tail)

  it('a refresh after compaction over a window that predates it', () => {
    // The held window starts inside the refreshed page: the page's first row is
    // not held, so the graft merges by row id, where the carried rows differ.
    const merged = graftRefreshedTailOntoBackfill(hydrate(pages.afterTurn.all), before)

    expectOneEntryPerMessage(merged, pages.afterTurn.all)
    expect(texts(merged)).toEqual(texts(hydrate(pages.afterTurn.all)))
  })

  it('a reload after compaction, then a backfill page read before it', () => {
    const reloaded = graftRefreshedTailOntoBackfill(hydrate(pages.afterCompaction.tail), before)
    const merged = mergeOlderTranscriptPage(reloaded, hydrate(pages.before.all))

    expectOneEntryPerMessage(merged, pages.afterCompaction.all)
  })

  it('a backfill page read after compaction, applied before the refresh', () => {
    const backfilled = mergeOlderTranscriptPage(before, hydrate(pages.afterCompaction.all))
    const refreshed = graftRefreshedTailOntoBackfill(hydrate(pages.afterTurn.tail), backfilled)

    expectOneEntryPerMessage(backfilled, pages.afterCompaction.all)
    expectOneEntryPerMessage(refreshed, pages.afterTurn.all)
  })

  it('a backfill page read before compaction, applied after the refresh', () => {
    const refreshed = graftRefreshedTailOntoBackfill(hydrate(pages.afterTurn.all), before)
    const merged = mergeOlderTranscriptPage(refreshed, hydrate(pages.before.tail))

    expectOneEntryPerMessage(merged, pages.afterTurn.all)
  })

  it('the /compress result (gateway rows, no uid) merged over the held window', () => {
    // slash.ts: mergeOlderTranscriptPage(post-compress history, held window).
    const merged = mergeOlderTranscriptPage(gatewayHydrate(pages.afterCompaction.all), before)

    expectOneEntryPerMessage(merged, pages.afterCompaction.all)
  })

  it('heals a window that already holds both copies', () => {
    const doubled = [...hydrate(pages.before.all), ...hydrate(pages.afterCompaction.tail).slice(-4)]
    const merged = graftRefreshedTailOntoBackfill(hydrate(pages.afterTurn.all), doubled)

    expectOneEntryPerMessage(merged, pages.afterTurn.all)
  })

  it('moves a copy to the newest row id and leaves other messages and arrays alone', () => {
    const fresh = hydrate(pages.afterCompaction.tail)
    const [, held] = unifyCompactionCopies(fresh, before)

    expect(held.map(message => message.rowId)).toEqual([17, 18, 19, 20, 26, 27, 28, 29])
    expect(held[0]).toBe(before[0])

    const unrelated = hydrate(pages.before.older)
    const [a, b] = unifyCompactionCopies(unrelated, before)

    expect(a).toBe(unrelated)
    expect(b).toBe(before)
  })
})
