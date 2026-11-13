import { describe, expect, it, vi } from 'vitest'

import { type ChatMessage, toChatMessages } from '@/lib/chat-messages'
import type { SessionMessage } from '@/types/hermes'

import {
  extendRefreshPageToOverlap,
  graftRefreshedTailOntoBackfill,
  mergeOlderTranscriptPage,
  unifyCompactionCopies
} from './transcript-backfill'

// Where the compaction-copy match must NOT fire, and how it meets the uid
// overlap anchor.
//
// The text key (role, timestamp, text) stands in for the uid only where there is
// none to read. The guard is the one PR #126834 settled on for its
// `logicalRowKey` (text-only rows; its transcript-backfill.tool-row-identity
// test is the second case here, rebuilt from stored rows), with the uid ahead of
// it. The overlap anchor and the folded tool message's uid come from PR #127685;
// the last block checks the graft that runs after that anchor.

const stored = (
  id: number,
  role: 'assistant' | 'user',
  content: string,
  timestamp: number,
  extra: Partial<SessionMessage> = {}
): SessionMessage => ({ content, id, role, timestamp, ...extra })

const toolCall = (id: string) => ({ function: { arguments: '{}', name: 'read_file' }, id })

const rowIds = (messages: ChatMessage[]) => messages.map(message => message.rowId)

const chat = (id: string, rowId?: number, messageUid?: string): ChatMessage => ({
  id,
  parts: [{ text: id, type: 'text' }],
  role: 'user',
  ...(rowId !== undefined ? { rowId } : {}),
  ...(messageUid !== undefined ? { messageUid } : {})
})

describe('the text key yields to the uid', () => {
  it('keeps two rows with the same role, timestamp and text when their uids differ', () => {
    const held = toChatMessages([
      stored(1, 'user', 'go', 1, { message_uid: 'uid-1' }),
      stored(2, 'assistant', 'ok', 5, { message_uid: 'uid-2' })
    ])

    const page = toChatMessages([
      stored(1, 'user', 'go', 1, { message_uid: 'uid-1' }),
      stored(2, 'assistant', 'ok', 5, { message_uid: 'uid-2' }),
      stored(3, 'user', 'again', 5, { message_uid: 'uid-3' }),
      stored(4, 'assistant', 'ok', 5, { message_uid: 'uid-4' })
    ])

    const [a, b] = unifyCompactionCopies(held, page)

    expect(a).toBe(held)
    expect(b).toBe(page)
    expect(rowIds(mergeOlderTranscriptPage(held, page))).toEqual([1, 2, 3, 4])
    expect(rowIds(graftRefreshedTailOntoBackfill(page, held))).toEqual([1, 2, 3, 4])
  })

  it('does not guess for a row without a uid when two uids share its text key', () => {
    const held = toChatMessages([
      stored(2, 'assistant', 'ok', 5, { message_uid: 'uid-2' }),
      stored(4, 'assistant', 'ok', 5, { message_uid: 'uid-4' })
    ])

    const gatewayRow = toChatMessages([{ content: null, role: 'assistant', row_id: 9, text: 'ok', timestamp: 5 }])
    const [a, b] = unifyCompactionCopies(held, gatewayRow)

    expect(a).toBe(held)
    expect(b).toBe(gatewayRow)
  })

  it('still matches a row without a uid to the one copy that has one', () => {
    const held = toChatMessages([stored(2, 'assistant', 'ok', 5, { message_uid: 'uid-2' })])
    const gatewayRow = toChatMessages([{ content: null, role: 'assistant', row_id: 9, text: 'ok', timestamp: 5 }])
    const [a] = unifyCompactionCopies(held, gatewayRow)

    expect(rowIds(a)).toEqual([9])
  })
})

describe('the text key is for rows that are only text', () => {
  it('keeps a tool-bearing reply distinct from a plain reply with the same text and timestamp', () => {
    const previous = toChatMessages([
      stored(2, 'user', 'anchor', 2),
      stored(3, 'assistant', 'answer', 20, { tool_calls: [toolCall('call-3')] })
    ])

    const refreshed = toChatMessages([
      stored(1, 'user', 'before', 1),
      stored(2, 'user', 'anchor', 2),
      stored(4, 'assistant', 'answer', 20)
    ])

    const merged = graftRefreshedTailOntoBackfill(refreshed, previous)

    expect(rowIds(merged)).toEqual([1, 2, 3, 4])
    expect(merged[2].parts.some(part => part.type === 'tool-call')).toBe(true)
    // The older-page merge places an unheld row behind its last anchor; what
    // matters here is that both rows are still there.
    expect(new Set(rowIds(mergeOlderTranscriptPage(previous, refreshed)))).toEqual(new Set([1, 2, 3, 4]))
  })

  it('still unifies tool-bearing copies by uid', () => {
    const before = toChatMessages([
      stored(3, 'assistant', 'answer', 20, { message_uid: 'uid-3', tool_calls: [toolCall('call-3')] })
    ])

    const after = toChatMessages([
      stored(8, 'assistant', 'answer', 20, { message_uid: 'uid-3', tool_calls: [toolCall('call-3')] })
    ])

    const [held] = unifyCompactionCopies(before, after)

    expect(rowIds(held)).toEqual([8])
  })

  it('still unifies a reply that carries reasoning: the backend row key has no reasoning in it', () => {
    const before = toChatMessages([
      { content: null, reasoning: 'thinking', role: 'assistant', row_id: 3, text: 'answer', timestamp: 20 }
    ])

    const after = toChatMessages([
      { content: null, reasoning: 'thinking', role: 'assistant', row_id: 8, text: 'answer', timestamp: 20 }
    ])

    const [held] = unifyCompactionCopies(before, after)

    expect(rowIds(held)).toEqual([8])
  })
})

describe('after the uid overlap anchor (PR #127685)', () => {
  it('grafts a page anchored by uid without keeping both copies', async () => {
    const previous = [chat('earlier', 1, 'uid-1'), chat('prompt', 2, 'uid-2'), chat('reply', 3, 'uid-3')]
    const refreshedTail = [chat('prompt', 102, 'uid-2'), chat('reply', 103, 'uid-3')]
    const readOlderPage = vi.fn().mockResolvedValue([])

    const extended = await extendRefreshPageToOverlap(refreshedTail, previous, readOlderPage)

    expect(extended).toBe(refreshedTail)
    expect(readOlderPage).not.toHaveBeenCalled()
    // And the graft that follows keeps the earlier row once, the copies once.
    expect(rowIds(graftRefreshedTailOntoBackfill(extended, previous))).toEqual([1, 102, 103])
  })

  it('carries the first uid of a folded tool batch onto the folded message', () => {
    const folded = toChatMessages([
      stored(4, 'user', 'do it', 1, { message_uid: 'uid-4' }),
      { content: null, id: 5, message_uid: 'uid-5', role: 'assistant', timestamp: 2, tool_calls: [toolCall('call-5')] },
      { content: 'done', id: 6, message_uid: 'uid-6', role: 'tool', timestamp: 3, tool_call_id: 'call-5' }
    ])

    expect(folded.map(message => message.messageUid)).toEqual(['uid-4', 'uid-5'])
  })
})
