import { describe, expect, it } from 'vitest'

import { graftRefreshedTailOntoBackfill } from '@/app/chat/transcript-backfill'
import { type ChatMessage, toChatMessages } from '@/lib/chat-messages'
import type { SessionMessage } from '@/types/hermes'

/**
 * A refresh reads only the newest page of a long session, and that page can
 * begin mid tool batch. `toChatMessages` flushes such a batch as a synthetic
 * assistant message whose id is not durable, so the page's first rendered row
 * has no `rowId`. `graftRefreshedTailOntoBackfill` used to read that
 * unanchored row as proof of earlier history and re-prepend it on every
 * refresh, so the window grew a duplicate fold per read (#123856).
 */

const userRow = (rowId: number, text: string): SessionMessage =>
  ({ id: rowId, role: 'user', content: text, timestamp: 1_000 + rowId }) as SessionMessage

const assistantRow = (rowId: number, text: string): SessionMessage =>
  ({ id: rowId, role: 'assistant', content: text, timestamp: 1_000 + rowId }) as SessionMessage

/** A standalone tool row: `toChatMessages` folds it into a message with no rowId. */
const toolRow = (rowId: number, text: string): SessionMessage =>
  ({ id: rowId, role: 'tool', content: text, timestamp: 1_000 + rowId }) as SessionMessage

/** The newest page opens on a tool row with no assistant answer under it yet. */
const pageOpeningOnToolRow = (): SessionMessage[] => [
  toolRow(100, 'reading files'),
  userRow(101, 'next question'),
  assistantRow(102, 'next answer'),
  userRow(103, 'final question'),
  assistantRow(104, 'final answer')
]

describe('refresh graft over a page that opens on a tool fold', () => {
  it('is a fixed point when the window was hydrated from that same page', () => {
    const page = pageOpeningOnToolRow()
    const local = toChatMessages(page)
    const remote = toChatMessages(page)

    // The page's leading fold is already in the window: nothing is missing, so
    // the graft must not manufacture an extra row.
    expect(local[0].rowId).toBeUndefined()
    expect(graftRefreshedTailOntoBackfill(remote, local)).toEqual(remote)
  })

  it('still keeps a backfilled prefix that is actually earlier than the page', () => {
    const page = pageOpeningOnToolRow()
    const olderPrefix = toChatMessages([userRow(1, 'ancient'), assistantRow(2, 'ancient reply')])
    const local = [...olderPrefix, ...toChatMessages(page)]

    const grafted = graftRefreshedTailOntoBackfill(toChatMessages(page), local)

    expect(grafted.length).toBe(local.length)
    expect(grafted[0]).toEqual(olderPrefix[0])
  })

  it('stays stable across repeated refreshes over a backfilled prefix', () => {
    const page = pageOpeningOnToolRow()
    const olderPrefix = toChatMessages([userRow(1, 'ancient'), assistantRow(2, 'ancient reply')])
    const local = [...olderPrefix, ...toChatMessages(page)]

    let grafted = local

    for (let round = 0; round < 5; round += 1) {
      grafted = graftRefreshedTailOntoBackfill(toChatMessages(page), grafted)

      expect(grafted.length).toBe(local.length)
      expect(grafted[0]).toEqual(olderPrefix[0])
    }
  })

  it('still reads an unstored fold the page does not carry as earlier history', () => {
    const page = pageOpeningOnToolRow()

    // A fold the refreshed page has no copy of: it can only have come from an
    // older page, so it stays in front of the refreshed tail.
    const olderFold: ChatMessage = {
      id: '900-0-tools',
      parts: [{ completedAt: 900, type: 'tool-call', toolCallId: 'old-call', toolName: 'tool' }],
      role: 'assistant'
    }

    const local = [olderFold, ...toChatMessages(page)]

    const grafted = graftRefreshedTailOntoBackfill(toChatMessages(page), local)

    // The older fold survives in front of the page; the fold the page already
    // carries is not duplicated.
    expect(grafted.length).toBe(local.length)
    expect(grafted[0]).toEqual(olderFold)
  })
})
