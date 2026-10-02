import { describe, expect, it } from 'vitest'

import { type ChatMessage, textPart } from '@/lib/chat-messages'

import { graftRefreshedTailOntoBackfill } from './transcript-backfill'

/**
 * #126229: turns render out of order after a compaction handoff chain.
 *
 * A compression handoff ends each parent session with
 * `end_reason='compression'` and re-inserts the carried tail into the child
 * under FRESH row ids (the clone keeps the original rows' timestamps, so the
 * store's display dedupe prefers the fresh generation as the representative).
 * The lineage read the desktop refreshes with is CORRECT — strictly monotonic
 * ids, each logical row once — but it re-addresses rows the rendered window
 * already holds: the page carries id 12 where the window holds id 8 for the
 * same logical row.
 *
 * The numbers below come from a real SessionDB run of that shape (root ->
 * seg2 -> seg3, each parent ended 'compression'):
 *
 *   seg1 rows 1-4: u1 a1 u2 a2
 *   seg2 rows 5-10: SUMMARY(5), clones a1(6) u2(7) a2(8), live u3(9) a3(10)
 *   seg3 rows 11-16: SUMMARY(11), clones a2(12) u3(13) a3(14), u4(15) a4(16)
 *
 * seg3's lineage page reads ids [1, 6, 7, 12, 5, 13, 14, 11, 15, 16]: the
 * a2/u3/a3 tail is now addressed by its FRESH seg3 clones (12/13/14), not the
 * seg2 ids the window holds (8/9/10).
 */
const bubble = (role: 'user' | 'assistant', text: string, rowId?: number, extra?: Partial<ChatMessage>): ChatMessage => ({
  id: `r${rowId ?? 'live'}-${role}-${text}`,
  role,
  parts: [textPart(text)],
  ...(rowId !== undefined ? { rowId } : {}),
  ...extra
})

// The window as hydrated during seg2 (it rendered the same lineage then):
// rows 5/6/7 are shared with the refreshed page, rows 8/9/10 are the rows the
// handoff chain re-id'd, and the u4 turn streamed live across the handoff
// (the WebSocket dropped and reconnected before its completion arrived).
const seg2Window: ChatMessage[] = [
  bubble('user', '[CONTEXT COMPACTION - REFERENCE ONLY] S1', 5),
  bubble('assistant', 'a1', 6),
  bubble('user', 'u2', 7),
  bubble('assistant', 'a2', 8),
  bubble('user', 'u3', 9),
  bubble('assistant', 'a3', 10),
  bubble('user', 'u4', undefined, { id: 'user-1790000000000-ab' }),
  bubble('assistant', 'a4', undefined, { id: 'assistant-stream-1790000000000-0', pending: true })
]

// The refreshed page after the handoff chain (seg3's lineage latest page).
// Its first durable row (1) is not in the window, so the graft cannot splice
// an older prefix and falls through to the stored-id merge.
const seg3Page: ChatMessage[] = [
  bubble('user', 'u1', 1),
  bubble('assistant', 'a1', 6),
  bubble('user', 'u2', 7),
  bubble('assistant', 'a2', 12),
  bubble('user', '[CONTEXT COMPACTION - REFERENCE ONLY] S1', 5),
  bubble('user', 'u3', 13),
  bubble('assistant', 'a3', 14),
  bubble('user', '[CONTEXT COMPACTION - REFERENCE ONLY] S2', 11),
  bubble('user', 'u4', 15),
  bubble('assistant', 'a4', 16)
]

const turnTexts = (messages: ChatMessage[]): string[] =>
  messages
    .map(message => (message.parts[0].type === 'text' ? message.parts[0].text : ''))
    .filter(text => !text.startsWith('[CONTEXT COMPACTION'))

describe('graftRefreshedTailOntoBackfill / compaction handoff re-id (#126229)', () => {
  it('renders each logical turn once, in stored order, after a handoff chain re-ids the tail', () => {
    const merged = graftRefreshedTailOntoBackfill(seg3Page, seg2Window)

    // The page is authoritative: u1 a1 u2 a2 (S1) u3 a3 (S2) u4 a4, each
    // exactly once. The window's stale generations (8/9/10) must not paint
    // beside their re-id'd representatives (12/13/14), and the stranded live
    // copies of the committed u4/a4 turn must not stay pinned below it.
    expect(turnTexts(merged)).toEqual(['u1', 'a1', 'u2', 'a2', 'u3', 'a3', 'u4', 'a4'])
  })

  it('keeps an unsettled live reply the page does not carry yet', () => {
    // The next turn is still streaming: the page has not committed u5/a5, so
    // the live rows are the only copy and must survive at the tail.
    const window = seg2Window.slice(0, 6).concat([
      bubble('user', 'u5', undefined, { id: 'user-1790000005000-cd' }),
      bubble('assistant', 'streaming a5', undefined, { id: 'assistant-stream-1790000005000-1', pending: true })
    ])

    const merged = graftRefreshedTailOntoBackfill(seg3Page, window)

    expect(merged.slice(-2).map(m => m.id)).toEqual(['user-1790000005000-cd', 'assistant-stream-1790000005000-1'])
  })

  it('keeps a still-streaming reply when the page row is only a partial commit', () => {
    // The page's u4/a4 turn is a tool fold (durableComplete false): the turn
    // may still be running server-side, so the live stream copy stays.
    const partialPage = seg3Page.map(message =>
      message.rowId === 16 ? { ...message, durableComplete: false as const } : message
    )

    const merged = graftRefreshedTailOntoBackfill(partialPage, seg2Window)

    expect(merged.map(m => m.id)).toContain('assistant-stream-1790000000000-0')
  })

  it('is idempotent when the same page refreshes again', () => {
    const once = graftRefreshedTailOntoBackfill(seg3Page, seg2Window)
    const twice = graftRefreshedTailOntoBackfill(seg3Page, once)

    expect(turnTexts(twice)).toEqual(turnTexts(once))
    expect([...twice].sort((a, b) => (a.rowId ?? -1) - (b.rowId ?? -1))).toEqual(
      [...once].sort((a, b) => (a.rowId ?? -1) - (b.rowId ?? -1))
    )
  })

  it('does not retire an older backfilled row that merely repeats a newer turn', () => {
    // A backfilled prefix row absent from the latest page is real history:
    // its id predates every row both sides share, so it must survive even
    // though a NEWER turn repeats its prose.
    const window = [
      bubble('user', 'ok', 20),
      bubble('assistant', 'sure', 21),
      bubble('user', 'ok', 40),
      bubble('assistant', 'sure', 41)
    ]

    const page = [bubble('user', 'ok', 40), bubble('assistant', 'sure', 41)]

    const merged = graftRefreshedTailOntoBackfill(page, window)

    expect(turnTexts(merged)).toEqual(['ok', 'sure', 'ok', 'sure'])
  })
})
