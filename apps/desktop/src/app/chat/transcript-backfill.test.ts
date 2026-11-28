import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { transcriptRowIds } from '@/app/session/hooks/use-session-actions/pending-turn-identity'
import { assistantTextPart, type ChatMessage, chatMessageText, toChatMessages } from '@/lib/chat-messages'
import { $transcriptTailBySessionId, recordTranscriptTail, transcriptTailState } from '@/store/transcript-tail'

import {
  _resetTranscriptBackfillForTests,
  backfillOlderTranscriptPage,
  extendRefreshPageToOverlap,
  graftRefreshedTailOntoBackfill,
  mergeOlderTranscriptPage,
  transcriptBackfillAvailable,
  unhideOpeningUserRows
} from './transcript-backfill'

vi.mock('@/hermes', () => ({
  getOlderSessionMessages: vi.fn(),
  getSessionMessages: vi.fn()
}))

const { getOlderSessionMessages, getSessionMessages } = await import('@/hermes')

const chat = (id: string, rowId?: number): ChatMessage => ({
  id,
  role: 'user',
  parts: [{ type: 'text', text: id }],
  ...(rowId !== undefined ? { rowId } : {})
})

// A stored SessionMessage row: distinct timestamps keep toChatMessages ids
// unique and the row id survives as ChatMessage.rowId.
const row = (rowId: number, text: string) => ({
  id: rowId,
  role: 'user' as const,
  content: text,
  timestamp: 1_000 + rowId
})

describe('transcript tail bookkeeping', () => {
  beforeEach(() => {
    $transcriptTailBySessionId.set({})
  })

  it('marks a full page as possibly truncated with the next offset', () => {
    recordTranscriptTail(
      'stored-1',
      {
        messages: Array.from({ length: 120 }, (_, index) => row(index + 500, `m${index}`)),
        pagination: { limit: 120, offset: 0, order: 'latest', returned: 120 }
      },
      'work'
    )

    expect(transcriptTailState('stored-1')).toEqual({ nextOffset: 120, possiblyTruncated: true, profile: 'work' })
    expect(transcriptBackfillAvailable('stored-1')).toBe(true)
  })

  it('marks a short page as complete', () => {
    recordTranscriptTail('stored-1', {
      messages: [row(1, 'only')],
      pagination: { limit: 120, offset: 0, order: 'latest', returned: 1 }
    })

    expect(transcriptBackfillAvailable('stored-1')).toBe(false)
  })

  it('treats a legacy response without pagination metadata as complete', () => {
    recordTranscriptTail('stored-1', {
      messages: Array.from({ length: 700 }, (_, index) => row(index, `m${index}`))
    })

    expect(transcriptBackfillAvailable('stored-1')).toBe(false)
  })

  it('counts the same session id on separate connections as separate entries', () => {
    const page = {
      messages: [row(1, 'tail')],
      pagination: { limit: 120, offset: 0, order: 'latest' as const, returned: 1 }
    }

    const sourceA = { connectionId: 'source-a', profile: 'backend' }
    const sourceB = { connectionId: 'source-b', profile: 'backend' }

    recordTranscriptTail('same-session', page, sourceA)
    recordTranscriptTail('same-session', page, sourceB)

    expect(Object.keys($transcriptTailBySessionId.get())).toHaveLength(2)
    expect(transcriptTailState('same-session', sourceA)?.profile).toEqual(sourceA)
    expect(transcriptTailState('same-session', sourceB)?.profile).toEqual(sourceB)
    expect(transcriptTailState('same-session')).toBeUndefined()
  })

  it('bounds entries and deterministically evicts the oldest scoped identity', () => {
    const page = {
      messages: [row(1, 'tail')],
      pagination: { limit: 120, offset: 0, order: 'latest' as const, returned: 1 }
    }

    const scope = { connectionId: 'source-a', profile: 'backend' }

    for (let index = 0; index < 257; index += 1) {
      recordTranscriptTail(`bounded-${index}`, page, scope)
    }

    expect(Object.keys($transcriptTailBySessionId.get())).toHaveLength(256)
    expect(transcriptTailState('bounded-0', scope)).toBeUndefined()
    expect(transcriptTailState('bounded-1', scope)).toBeDefined()
    expect(transcriptTailState('bounded-256', scope)).toBeDefined()
  })
})

describe('mergeOlderTranscriptPage', () => {
  it('prepends the older page and preserves chronological order', () => {
    const existing = [chat('c', 3), chat('d', 4)]
    const older = [chat('a', 1), chat('b', 2)]

    expect(mergeOlderTranscriptPage(existing, older).map(m => m.id)).toEqual(['a', 'b', 'c', 'd'])
  })

  it('dedupes rows the store already holds by durable row id', () => {
    const existing = [chat('b', 2), chat('c', 3)]
    // Offset drift: the fetched page overlaps one row we already have.
    const older = [chat('a', 1), chat('b-refetched', 2)]

    expect(mergeOlderTranscriptPage(existing, older).map(m => m.rowId)).toEqual([1, 2, 3])
  })

  it('keeps newer rows after an old tail when a drifting offset returns both overlap and subsequent turns', () => {
    // A page initially ending at row 6 was cached. New turns persisted before
    // the older-page request, so its offset now lands across rows 4–8.
    const existing = [
      chat('one', 1),
      chat('two', 2),
      chat('three', 3),
      chat('four', 4),
      chat('five', 5),
      chat('six', 6)
    ]

    const fetched = [
      chat('four-refetched', 4),
      chat('five-refetched', 5),
      chat('six-refetched', 6),
      chat('seven', 7),
      chat('eight', 8)
    ]

    const merged = mergeOlderTranscriptPage(existing, fetched)

    expect(merged.map(message => message.rowId)).toEqual([1, 2, 3, 4, 5, 6, 7, 8])
    expect(merged.slice(0, 6)).toEqual(existing)
    expect(mergeOlderTranscriptPage(merged, fetched)).toBe(merged)
  })

  it('keeps reference identity when every older row is already present', () => {
    const existing = [chat('a', 1), chat('b', 2)]
    const older = [chat('a', 1)]

    expect(mergeOlderTranscriptPage(existing, older)).toBe(existing)
  })

  it('refuses to paint an older page as the whole transcript', () => {
    const existing: ChatMessage[] = []

    expect(mergeOlderTranscriptPage(existing, [chat('a', 1)])).toBe(existing)
  })

  // Hydration folds a turn into one bubble addressed by its FIRST source row,
  // with the final reply (52203) riding inside as a text part. A drifting
  // older-page fetch that overlaps the just-persisted turn must not paint that
  // reply twice, but must not drop rows the store does not hold either.
  const narration = { type: 'text', text: 'earlier tool narration', sourceRowId: 52201 } as const
  const answer = { type: 'text', text: 'the answer', sourceRowId: 52203 } as const

  const toolCall = {
    type: 'tool-call',
    toolCallId: 't1',
    toolName: 'read_file',
    args: {}
  } as unknown as ChatMessage['parts'][number]

  it.each([
    {
      name: 'drops a fold whose every row is already held live',
      live: { parts: [narration, answer], rowId: 52203 },
      foldParts: [narration, answer],
      expected: ['prompt', 'assistant-stream-1']
    },
    {
      name: 'keeps a fold carrying a tool call and unheld narration',
      live: { parts: [answer], rowId: 52203 },
      foldParts: [toolCall, narration, answer],
      expected: ['prompt', '1770000000000-4-assistant', 'assistant-stream-1']
    },
    {
      name: 'keeps a fold whose first row is held only as a part of another bubble',
      live: { parts: [narration], rowId: 52202 },
      foldParts: [narration, answer],
      expected: ['prompt', '1770000000000-4-assistant', 'assistant-stream-1']
    }
  ])('$name (#123801)', ({ live, foldParts, expected }) => {
    const existing: ChatMessage[] = [
      chat('prompt', 52200),
      { id: 'assistant-stream-1', role: 'assistant', ...live, parts: [...live.parts] }
    ]

    const fold: ChatMessage = {
      id: '1770000000000-4-assistant',
      role: 'assistant',
      parts: [...foldParts],
      rowId: 52201
    }

    expect(
      mergeOlderTranscriptPage(existing, [chat('prompt-refetch', 52200), fold]).map(message => message.id)
    ).toEqual(expected)
  })
})

describe('graftRefreshedTailOntoBackfill', () => {
  it('keeps the backfilled prefix when the refreshed tail anchors inside it', () => {
    const previous = [chat('a', 1), chat('b', 2), chat('c', 3)]
    const refreshed = [chat('b', 2), chat('c', 3), chat('d', 4)]

    expect(graftRefreshedTailOntoBackfill(refreshed, previous).map(m => m.rowId)).toEqual([1, 2, 3, 4])
  })

  it('returns the refreshed tail unchanged when no anchor is found', () => {
    const previous = [chat('x', 90), chat('y', 91), chat('z', 92)]
    const refreshed = [chat('p', 200), chat('q', 201)]

    expect(graftRefreshedTailOntoBackfill(refreshed, previous)).toBe(refreshed)
  })

  it('moves an older row that was glued on after the tail back to stored order', () => {
    const previous = [chat('tail-a', 700), chat('tail-b', 701), chat('tail-c', 746), chat('kcsie', 662)]
    const refreshed = [chat('kcsie', 662), chat('tail-a', 700), chat('tail-b', 701), chat('tail-c', 746)]

    expect(graftRefreshedTailOntoBackfill(refreshed, previous).map(message => message.rowId)).toEqual([
      662, 700, 701, 746
    ])

    // A page that starts mid-turn opens with a tool fold that has no stored
    // id. It stays in front of the stored row it preceded.
    const behindEarlier = [chat('earlier', 500), ...previous]
    const withFold = [chat('fold-tools'), ...refreshed, chat('next', 747)]

    expect(graftRefreshedTailOntoBackfill(withFold, behindEarlier).map(message => message.id)).toEqual([
      'earlier',
      'fold-tools',
      'kcsie',
      'tail-a',
      'tail-b',
      'tail-c',
      'next'
    ])
  })

  it('keeps the fresh page copy when the same stored id is on both sides', () => {
    const previous = [chat('tail-a', 700), chat('stale-b', 701), chat('tail-c', 746)]
    const refreshed = [chat('kcsie', 662), chat('fresh-b', 701), chat('tail-c', 746)]

    const merged = graftRefreshedTailOntoBackfill(refreshed, previous)

    expect(merged.map(message => message.rowId)).toEqual([662, 700, 701, 746])
    expect(merged.find(message => message.rowId === 701)).toBe(refreshed[1])
  })

  it('keeps the earlier transcript when a page-local fold precedes a shared durable row', () => {
    const previous = [chat('earlier', 1), chat('prompt', 2), chat('reply', 3)]
    const refreshed = [chat('page-local-fold'), chat('reply-refetched', 3), chat('new-reply', 4)]

    expect(graftRefreshedTailOntoBackfill(refreshed, previous).map(m => m.rowId)).toEqual([1, 2, undefined, 3, 4])
  })

  it('does not treat a page that opens on its own orphan tool fold as behind itself (#124311)', () => {
    const page = [chat('orphan-tools'), chat('user', 122886), chat('assistant', 122887)]

    expect(graftRefreshedTailOntoBackfill(page, page)).toBe(page)
  })

  it('returns the refreshed tail when it is not shorter than the previous transcript', () => {
    const previous = [chat('a', 1)]
    const refreshed = [chat('a', 1), chat('b', 2)]

    expect(graftRefreshedTailOntoBackfill(refreshed, previous)).toBe(refreshed)
  })

  it('keeps an unpersisted stream bubble when the refresh page omits the final assistant row', () => {
    const toolTurnRows = [
      { id: 10, role: 'user' as const, content: 'What is the status?', timestamp: 1 },
      {
        id: 11,
        role: 'assistant' as const,
        content: '',
        timestamp: 2,
        tool_calls: [
          { id: 'call-a', type: 'function' as const, function: { name: 'terminal', arguments: '{}' } }
        ]
      },
      {
        id: 12,
        role: 'tool' as const,
        content: '{"output": "ok", "exit_code": 0}',
        timestamp: 3,
        tool_call_id: 'call-a',
        tool_name: 'terminal'
      }
    ]

    const partialPage = toChatMessages(toolTurnRows)
    const liveReply = 'Live check, just now — visible while streaming'

    const livePrevious: ChatMessage[] = [
      { id: 'user-live', role: 'user', parts: [{ type: 'text', text: 'What is the status?' }], rowId: 10 },
      {
        id: 'assistant-stream-live',
        role: 'assistant',
        parts: [
          { type: 'tool-call', toolCallId: 'call-a', toolName: 'terminal', args: '{}' },
          assistantTextPart(liveReply, 1)
        ],
        durableComplete: false
      }
    ]

    const merged = graftRefreshedTailOntoBackfill(partialPage, livePrevious)
    const userIndex = merged.findIndex(message => message.rowId === 10)

    const replyOnTurn = merged.find(
      (message, index) =>
        index > userIndex && message.role === 'assistant' && !message.hidden && chatMessageText(message).trim().length > 0
    )

    expect(replyOnTurn).toBeDefined()
    expect(chatMessageText(replyOnTurn!).trim()).toContain('Live check')
    expect(replyOnTurn!.parts.some(part => part.type === 'tool-call' && part.toolCallId === 'call-a')).toBe(true)
    expect(merged.filter(message => message.role === 'assistant' && chatMessageText(message).includes('Live check'))).toHaveLength(
      1
    )
  })

  it('keeps an unbound stream bubble without durableComplete when the page omits the final row', () => {
    const toolTurnRows = [
      { id: 10, role: 'user' as const, content: 'What is the status?', timestamp: 1 },
      {
        id: 11,
        role: 'assistant' as const,
        content: '',
        timestamp: 2,
        tool_calls: [
          { id: 'call-a', type: 'function' as const, function: { name: 'terminal', arguments: '{}' } }
        ]
      }
    ]

    const partialPage = toChatMessages(toolTurnRows)
    const liveReply = 'Live check, just now — visible while streaming'

    const livePrevious: ChatMessage[] = [
      { id: 'user-live', role: 'user', parts: [{ type: 'text', text: 'What is the status?' }], rowId: 10 },
      {
        id: 'assistant-stream-live',
        role: 'assistant',
        parts: [{ type: 'text', text: liveReply }]
      }
    ]

    expect(transcriptRowIds(livePrevious[1]!)).toEqual([])
    expect(livePrevious[1]?.durableComplete).toBeUndefined()

    const merged = graftRefreshedTailOntoBackfill(partialPage, livePrevious)
    const userIndex = merged.findIndex(message => message.rowId === 10)

    const replyOnTurn = merged.find(
      (message, index) =>
        index > userIndex && message.role === 'assistant' && !message.hidden && chatMessageText(message).trim().length > 0
    )

    expect(replyOnTurn).toBeDefined()
    expect(chatMessageText(replyOnTurn!).trim()).toContain('Live check')
  })

  it('keeps the grafted reply on a second refresh with the same partial page', () => {
    const toolTurnRows = [
      { id: 10, role: 'user' as const, content: 'What is the status?', timestamp: 1 },
      {
        id: 11,
        role: 'assistant' as const,
        content: '',
        timestamp: 2,
        tool_calls: [
          { id: 'call-a', type: 'function' as const, function: { name: 'terminal', arguments: '{}' } }
        ]
      }
    ]

    const partialPage = toChatMessages(toolTurnRows)

    const livePrevious: ChatMessage[] = [
      { id: 'user-live', role: 'user', parts: [{ type: 'text', text: 'What is the status?' }], rowId: 10 },
      {
        id: 'assistant-stream-live',
        role: 'assistant',
        parts: [assistantTextPart('Live check, just now', 1)],
        durableComplete: false
      }
    ]

    const first = graftRefreshedTailOntoBackfill(partialPage, livePrevious)
    const second = graftRefreshedTailOntoBackfill(partialPage, first)
    const userIndex = second.findIndex(message => message.rowId === 10)

    const replyOnTurn = second.find(
      (message, index) =>
        index > userIndex && message.role === 'assistant' && !message.hidden && chatMessageText(message).trim().length > 0
    )

    expect(replyOnTurn).toBeDefined()
    expect(chatMessageText(replyOnTurn!).trim()).toContain('Live check')
    expect(replyOnTurn?.durableComplete).toBe(false)
  })

  it('re-anchors the live reply on its user turn when the page adds a later user row', () => {
    const refreshedPage: ChatMessage[] = [
      {
        id: 'user-status',
        role: 'user',
        parts: [{ type: 'text', text: 'What is the status?' }],
        rowId: 10
      },
      {
        id: 'asst-tool',
        role: 'assistant',
        parts: [{ type: 'tool-call', toolCallId: 'call-a', toolName: 'terminal', args: '{}' }],
        rowId: 11
      },
      { id: 'user-later', role: 'user', parts: [{ type: 'text', text: 'Next prompt' }], rowId: 99 }
    ]

    const liveReply = 'Final table output for status'

    const livePrevious: ChatMessage[] = [
      { id: 'user-status', role: 'user', parts: [{ type: 'text', text: 'What is the status?' }], rowId: 10 },
      {
        id: 'assistant-stream-live',
        role: 'assistant',
        parts: [assistantTextPart(liveReply, 1)],
        durableComplete: false
      }
    ]

    const merged = graftRefreshedTailOntoBackfill(refreshedPage, livePrevious)
    const statusUserIdx = merged.findIndex(message => message.rowId === 10)
    const laterUserIdx = merged.findIndex(message => message.rowId === 99)

    const replyOnTurn = merged.find(
      (message, index) =>
        index > statusUserIdx &&
        index < laterUserIdx &&
        message.role === 'assistant' &&
        !message.hidden &&
        chatMessageText(message).trim().length > 0
    )

    expect(replyOnTurn).toBeDefined()
    expect(chatMessageText(replyOnTurn!).trim()).toBe(liveReply)
    expect(replyOnTurn!.parts.some(part => part.type === 'tool-call' && part.toolCallId === 'call-a')).toBe(true)
    expect(merged.filter(message => message.role === 'assistant' && chatMessageText(message) === liveReply)).toHaveLength(1)
  })

  it('keeps the prior transcript on an empty refresh only while an unpersisted live reply exists', () => {
    const livePrevious: ChatMessage[] = [
      { id: 'user', role: 'user', parts: [{ type: 'text', text: 'Hi' }], rowId: 1 },
      { id: 'assistant-live', role: 'assistant', parts: [assistantTextPart('visible reply', 1)], durableComplete: false }
    ]

    const populatedPrevious: ChatMessage[] = [
      { id: 'user', role: 'user', parts: [{ type: 'text', text: 'Hi' }], rowId: 1 },
      {
        id: 'assistant-done',
        role: 'assistant',
        parts: [assistantTextPart('done', 1)],
        rowId: 2,
        durableComplete: true
      }
    ]

    expect(graftRefreshedTailOntoBackfill([], livePrevious)).toBe(livePrevious)
    expect(graftRefreshedTailOntoBackfill([], populatedPrevious)).toEqual([])
  })

  it('does not re-add a live reply when the refresh page already carries the final row', () => {
    const liveReply = 'Final table output'

    const refreshedPage: ChatMessage[] = [
      { id: 'user-status', role: 'user', parts: [{ type: 'text', text: 'Status?' }], rowId: 10 },
      {
        id: 'asst-final',
        role: 'assistant',
        parts: [assistantTextPart(liveReply, 12)],
        rowId: 12
      }
    ]

    const previous: ChatMessage[] = [
      { id: 'user-status', role: 'user', parts: [{ type: 'text', text: 'Status?' }], rowId: 10 },
      {
        id: 'assistant-stream-live',
        role: 'assistant',
        parts: [assistantTextPart(liveReply, 1)],
        durableComplete: false
      }
    ]

    const merged = graftRefreshedTailOntoBackfill(refreshedPage, previous)

    expect(merged.filter(message => message.role === 'assistant' && chatMessageText(message) === liveReply)).toHaveLength(1)
  })

  it('anchors by user message id when repeated prompt text would match a later turn', () => {
    const refreshedPage: ChatMessage[] = [
      { id: 'user-first', role: 'user', parts: [{ type: 'text', text: 'yes' }], rowId: 1 },
      {
        id: 'asst-empty',
        role: 'assistant',
        parts: [{ type: 'tool-call', toolCallId: 'call-a', toolName: 'terminal', args: '{}' }],
        rowId: 2
      },
      { id: 'user-second', role: 'user', parts: [{ type: 'text', text: 'yes' }], rowId: 3 }
    ]

    const previous: ChatMessage[] = [
      { id: 'user-first', role: 'user', parts: [{ type: 'text', text: 'yes' }], rowId: 1 },
      {
        id: 'assistant-stream-live',
        role: 'assistant',
        parts: [assistantTextPart('Reply for the first yes', 1)],
        durableComplete: false
      }
    ]

    const merged = graftRefreshedTailOntoBackfill(refreshedPage, previous)
    const firstUserIdx = merged.findIndex(message => message.id === 'user-first')
    const secondUserIdx = merged.findIndex(message => message.id === 'user-second')
    const replyIdx = merged.findIndex(message => message.id === 'assistant-stream-live' || message.id === 'asst-empty')

    expect(replyIdx).toBeGreaterThan(firstUserIdx)
    expect(replyIdx).toBeLessThan(secondUserIdx)
    expect(chatMessageText(merged[replyIdx]).trim()).toBe('Reply for the first yes')
  })

  it('does not re-graft onto a settled turn after refresh', () => {
    const previous: ChatMessage[] = [
      { id: 'user-a', role: 'user', parts: [{ type: 'text', text: 'First' }], rowId: 1 },
      {
        id: 'assistant-long',
        role: 'assistant',
        parts: [assistantTextPart('A long completed answer on screen.', 1)],
        rowId: 2,
        durableComplete: true
      }
    ]

    const refreshed: ChatMessage[] = [
      ...previous,
      { id: 'user-b', role: 'user', parts: [{ type: 'text', text: 'Second' }], rowId: 3 },
      { id: 'assistant-short', role: 'assistant', parts: [assistantTextPart('ok', 1)], rowId: 4, durableComplete: true }
    ]

    const merged = graftRefreshedTailOntoBackfill(refreshed, previous)

    expect(merged.filter(message => message.rowId === 2)).toHaveLength(1)
    expect(merged[merged.length - 1]?.rowId).toBe(4)
  })

  it('does not duplicate when the page final row is a shorter suffix of the live bubble', () => {
    const narrationPlusFinal =
      'Step one: checking inputs. Step two: summarizing. Final table output for status.'

    const finalOnly = 'Final table output for status.'

    const refreshedPage: ChatMessage[] = [
      { id: 'user-status', role: 'user', parts: [{ type: 'text', text: 'Status?' }], rowId: 10 },
      {
        id: 'asst-tool',
        role: 'assistant',
        parts: [{ type: 'tool-call', toolCallId: 'call-a', toolName: 'terminal', args: '{}' }],
        rowId: 11,
        durableComplete: false
      },
      {
        id: 'asst-final',
        role: 'assistant',
        parts: [assistantTextPart(finalOnly, 12)],
        rowId: 12,
        durableComplete: true
      }
    ]

    const previous: ChatMessage[] = [
      { id: 'user-status', role: 'user', parts: [{ type: 'text', text: 'Status?' }], rowId: 10 },
      {
        id: 'assistant-stream-live',
        role: 'assistant',
        parts: [assistantTextPart(narrationPlusFinal, 1)],
        durableComplete: false
      }
    ]

    const merged = graftRefreshedTailOntoBackfill(refreshedPage, previous)
    const withText = merged.filter(message => message.role === 'assistant' && chatMessageText(message).includes(finalOnly))

    expect(withText).toHaveLength(1)
    expect(chatMessageText(withText[0]!).trim()).toBe(finalOnly)
  })

  it('converges to a single reply when the final row lands after a partial graft', () => {
    const partialPage: ChatMessage[] = [
      { id: 'user-status', role: 'user', parts: [{ type: 'text', text: 'Status?' }], rowId: 10 },
      {
        id: 'asst-tool',
        role: 'assistant',
        parts: [{ type: 'tool-call', toolCallId: 'call-a', toolName: 'terminal', args: '{}' }],
        rowId: 11,
        durableComplete: false
      }
    ]

    const finalPage: ChatMessage[] = [
      ...partialPage,
      {
        id: 'asst-final',
        role: 'assistant',
        parts: [assistantTextPart('Final table output', 12)],
        rowId: 12,
        durableComplete: true
      }
    ]

    const livePrevious: ChatMessage[] = [
      { id: 'user-status', role: 'user', parts: [{ type: 'text', text: 'Status?' }], rowId: 10 },
      {
        id: 'assistant-stream-live',
        role: 'assistant',
        parts: [assistantTextPart('Narration live. Final table output', 1)],
        durableComplete: false
      }
    ]

    const afterPartial = graftRefreshedTailOntoBackfill(partialPage, livePrevious)
    const afterFinal = graftRefreshedTailOntoBackfill(finalPage, afterPartial)

    const replies = afterFinal.filter(
      message => message.role === 'assistant' && chatMessageText(message).includes('Final table output')
    )

    expect(replies).toHaveLength(1)
    expect(replies[0]?.rowId).toBe(12)
  })

  it('maps repeated prompt text to the Nth user row from the end', () => {
    const refreshedPage: ChatMessage[] = [
      { id: 'u1', role: 'user', parts: [{ type: 'text', text: 'go' }] },
      { id: 'u2', role: 'user', parts: [{ type: 'text', text: 'go' }], rowId: 2 }
    ]

    const previous: ChatMessage[] = [
      { id: 'orphan-user', role: 'user', parts: [{ type: 'text', text: 'go' }] },
      {
        id: 'assistant-stream-live',
        role: 'assistant',
        parts: [assistantTextPart('orphan reply', 1)],
        durableComplete: false
      }
    ]

    const merged = graftRefreshedTailOntoBackfill(refreshedPage, previous)
    const u2Idx = merged.findIndex(message => message.rowId === 2)
    const replyIdx = merged.findIndex(message => chatMessageText(message).includes('orphan reply'))

    expect(replyIdx).toBeGreaterThan(u2Idx)
    expect(merged.filter(message => chatMessageText(message).includes('orphan reply'))).toHaveLength(1)
  })

  // Deliberate: we cannot tell a regenerate from a stale refresh page, so both replies may show.
  it('still grafts when the last persisted assistant text is not the live final', () => {
    const refreshedPage: ChatMessage[] = [
      { id: 'user', role: 'user', parts: [{ type: 'text', text: 'Again' }], rowId: 1 },
      {
        id: 'asst-old',
        role: 'assistant',
        parts: [assistantTextPart('Superseded answer.', 2)],
        rowId: 2,
        durableComplete: true
      }
    ]

    const previous: ChatMessage[] = [
      { id: 'user', role: 'user', parts: [{ type: 'text', text: 'Again' }], rowId: 1 },
      {
        id: 'assistant-stream-live',
        role: 'assistant',
        parts: [assistantTextPart('Replacement answer.', 1)],
        durableComplete: false
      }
    ]

    const merged = graftRefreshedTailOntoBackfill(refreshedPage, previous)

    expect(merged.filter(message => chatMessageText(message).includes('Replacement answer.'))).toHaveLength(1)
    expect(merged.filter(message => chatMessageText(message).includes('Superseded answer.'))).toHaveLength(1)
  })

  it('still grafts when a text-only persisted row is only a prefix of the live bubble', () => {
    const narration = 'Checking inputs. Final table output for status.'
    const commentary = 'Checking inputs.'

    const refreshedPage: ChatMessage[] = [
      { id: 'user-status', role: 'user', parts: [{ type: 'text', text: 'Status?' }], rowId: 10 },
      {
        id: 'asst-commentary',
        role: 'assistant',
        parts: [assistantTextPart(commentary, 11)],
        rowId: 11,
        durableComplete: true
      },
      {
        id: 'asst-tool',
        role: 'assistant',
        parts: [{ type: 'tool-call', toolCallId: 'call-a', toolName: 'terminal', args: '{}' }],
        rowId: 12,
        durableComplete: false
      }
    ]

    const previous: ChatMessage[] = [
      { id: 'user-status', role: 'user', parts: [{ type: 'text', text: 'Status?' }], rowId: 10 },
      {
        id: 'assistant-stream-live',
        role: 'assistant',
        parts: [assistantTextPart(narration, 1)],
        durableComplete: false
      }
    ]

    const merged = graftRefreshedTailOntoBackfill(refreshedPage, previous)
    const toolIdx = merged.findIndex(message => message.rowId === 12)

    expect(toolIdx).toBeGreaterThanOrEqual(0)
    expect(chatMessageText(merged[toolIdx]!).trim()).toContain('Final table output')
    expect(merged.filter(message => chatMessageText(message).includes('Checking inputs.'))).toHaveLength(1)
  })

  it('does not duplicate narration on a tool row that already holds the prefix text', () => {
    const commentary = 'Checking inputs.'
    const finalOnly = 'Final table output for status.'
    const liveNarration = `${commentary} ${finalOnly}`

    const refreshedPage: ChatMessage[] = [
      { id: 'user-status', role: 'user', parts: [{ type: 'text', text: 'Status?' }], rowId: 10 },
      {
        id: 'asst-narration-tool',
        role: 'assistant',
        parts: [assistantTextPart(commentary, 11), { type: 'tool-call', toolCallId: 'call-a', toolName: 'terminal', args: '{}' }],
        rowId: 11,
        durableComplete: false
      }
    ]

    const previous: ChatMessage[] = [
      { id: 'user-status', role: 'user', parts: [{ type: 'text', text: 'Status?' }], rowId: 10 },
      {
        id: 'assistant-stream-live',
        role: 'assistant',
        parts: [assistantTextPart(liveNarration, 1)],
        durableComplete: false
      }
    ]

    const merged = graftRefreshedTailOntoBackfill(refreshedPage, previous)
    const narrationRows = merged.filter(message => chatMessageText(message).includes(commentary))

    expect(narrationRows).toHaveLength(1)
    expect(chatMessageText(narrationRows[0]!).trim()).toContain(commentary)
    expect(chatMessageText(narrationRows[0]!).trim()).toContain(finalOnly)
  })

  it('keeps narration-tool graft on a second refresh with the same partial page', () => {
    const commentary = 'Checking inputs.'
    const finalOnly = 'Final table output for status.'
    const liveNarration = `${commentary} ${finalOnly}`

    const refreshedPage: ChatMessage[] = [
      { id: 'user-status', role: 'user', parts: [{ type: 'text', text: 'Status?' }], rowId: 10 },
      {
        id: 'asst-narration-tool',
        role: 'assistant',
        parts: [assistantTextPart(commentary, 11), { type: 'tool-call', toolCallId: 'call-a', toolName: 'terminal', args: '{}' }],
        rowId: 11,
        durableComplete: false
      }
    ]

    const livePrevious: ChatMessage[] = [
      { id: 'user-status', role: 'user', parts: [{ type: 'text', text: 'Status?' }], rowId: 10 },
      {
        id: 'assistant-stream-live',
        role: 'assistant',
        parts: [assistantTextPart(liveNarration, 1)],
        durableComplete: false
      }
    ]

    const first = graftRefreshedTailOntoBackfill(refreshedPage, livePrevious)
    const second = graftRefreshedTailOntoBackfill(refreshedPage, first)
    const narrationRows = second.filter(message => chatMessageText(message).includes(commentary))

    expect(narrationRows).toHaveLength(1)
    expect(chatMessageText(narrationRows[0]!).trim()).toContain(finalOnly)
  })

  it('re-grafts when a same-id row is shorter than the live text', () => {
    const liveReply = 'Checking inputs. Final table output for status.'

    const refreshedPage: ChatMessage[] = [
      { id: 'user-status', role: 'user', parts: [{ type: 'text', text: 'Status?' }], rowId: 10 },
      {
        id: 'asst-narration-tool',
        role: 'assistant',
        parts: [
          assistantTextPart('Checking inputs.', 11),
          { type: 'tool-call', toolCallId: 'call-a', toolName: 'terminal', args: '{}' }
        ],
        rowId: 11,
        durableComplete: false
      }
    ]

    const previous: ChatMessage[] = [
      { id: 'user-status', role: 'user', parts: [{ type: 'text', text: 'Status?' }], rowId: 10 },
      {
        id: 'asst-narration-tool',
        role: 'assistant',
        parts: [
          assistantTextPart('Checking inputs.', 11),
          { type: 'tool-call', toolCallId: 'call-a', toolName: 'terminal', args: '{}' },
          assistantTextPart('Final table output for status.', 1)
        ],
        rowId: 11,
        durableComplete: false
      }
    ]

    const merged = graftRefreshedTailOntoBackfill(refreshedPage, previous)

    expect(chatMessageText(merged.find(message => message.id === 'asst-narration-tool')!).trim()).toContain(
      'Final table output'
    )
  })

  it('does not treat a too-short suffix as a persisted final row', () => {
    const liveReply = 'We left the next day'
    const persistedSuffix = 'day'

    const refreshedPage: ChatMessage[] = [
      { id: 'user', role: 'user', parts: [{ type: 'text', text: 'Go' }], rowId: 1 },
      {
        id: 'asst-short',
        role: 'assistant',
        parts: [assistantTextPart(persistedSuffix, 2)],
        rowId: 2,
        durableComplete: true
      }
    ]

    const previous: ChatMessage[] = [
      { id: 'user', role: 'user', parts: [{ type: 'text', text: 'Go' }], rowId: 1 },
      {
        id: 'assistant-stream-live',
        role: 'assistant',
        parts: [assistantTextPart(liveReply, 1)],
        durableComplete: false
      }
    ]

    expect(graftRefreshedTailOntoBackfill(refreshedPage, previous).filter(message => chatMessageText(message) === liveReply)).toHaveLength(
      1
    )
  })

  it('does not treat a shared in-word suffix as a persisted final row', () => {
    const liveReply = 'Still shopping'
    const persistedSuffix = 'ping'

    const refreshedPage: ChatMessage[] = [
      { id: 'user', role: 'user', parts: [{ type: 'text', text: 'Go' }], rowId: 1 },
      {
        id: 'asst-inword',
        role: 'assistant',
        parts: [assistantTextPart(persistedSuffix, 2)],
        rowId: 2,
        durableComplete: true
      }
    ]

    const previous: ChatMessage[] = [
      { id: 'user', role: 'user', parts: [{ type: 'text', text: 'Go' }], rowId: 1 },
      {
        id: 'assistant-stream-live',
        role: 'assistant',
        parts: [assistantTextPart(liveReply, 1)],
        durableComplete: false
      }
    ]

    expect(graftRefreshedTailOntoBackfill(refreshedPage, previous).filter(message => chatMessageText(message) === liveReply)).toHaveLength(
      1
    )
  })

  it('does not re-graft when a same-id page row already covers the live text', () => {
    const fullReply = 'Checking inputs. Final table output for status.'

    const refreshedPage: ChatMessage[] = [
      { id: 'user-status', role: 'user', parts: [{ type: 'text', text: 'Status?' }], rowId: 10 },
      {
        id: 'asst-narration-tool',
        role: 'assistant',
        parts: [
          assistantTextPart(fullReply, 11),
          { type: 'tool-call', toolCallId: 'call-a', toolName: 'terminal', args: '{}' }
        ],
        rowId: 11,
        durableComplete: true
      }
    ]

    const previous: ChatMessage[] = [
      { id: 'user-status', role: 'user', parts: [{ type: 'text', text: 'Status?' }], rowId: 10 },
      {
        id: 'asst-narration-tool',
        role: 'assistant',
        parts: [
          assistantTextPart(fullReply, 11),
          { type: 'tool-call', toolCallId: 'call-a', toolName: 'terminal', args: '{}' }
        ],
        rowId: 11,
        durableComplete: false
      }
    ]

    const merged = graftRefreshedTailOntoBackfill(refreshedPage, previous)

    expect(merged[1]).toBe(refreshedPage[1])
    expect(merged[1]?.durableComplete).toBe(true)
  })

  it('does not re-graft a hydrated tool turn when the refresh page matches', () => {
    const finalText = 'All checks passed.'

    const rows = [
      { id: 1, role: 'user' as const, content: 'Run checks', timestamp: 1 },
      {
        id: 2,
        role: 'assistant' as const,
        content: '',
        timestamp: 2,
        tool_calls: [{ id: 'call-a', type: 'function' as const, function: { name: 'terminal', arguments: '{}' } }]
      },
      {
        id: 3,
        role: 'tool' as const,
        content: '{"output": "ok", "exit_code": 0}',
        timestamp: 3,
        tool_call_id: 'call-a',
        tool_name: 'terminal'
      },
      { id: 4, role: 'assistant' as const, content: finalText, timestamp: 4 }
    ]

    const page = toChatMessages(rows)

    const previous = page.map((message, index) =>
      index === page.length - 1 ? { ...message, durableComplete: false } : message
    )

    const merged = graftRefreshedTailOntoBackfill(page, previous)

    expect(merged).toBe(page)
    expect(merged.filter(message => message.role === 'assistant' && chatMessageText(message).includes(finalText))).toHaveLength(1)
  })
})

describe('extendRefreshPageToOverlap', () => {
  it('reads older pages until a long refresh shares a durable row with the rendered transcript', async () => {
    const previous = [chat('earlier', 1), chat('prompt', 2), chat('reply', 3)]
    const readOlderPage = vi.fn().mockResolvedValueOnce([chat('reply-refetched', 3), chat('tool-fold', 4)])

    const extended = await extendRefreshPageToOverlap(
      [chat('new-tool', 5), chat('new-reply', 6)],
      previous,
      readOlderPage
    )

    expect(readOlderPage).toHaveBeenCalledTimes(1)
    expect(graftRefreshedTailOntoBackfill(extended, previous).map(message => message.rowId)).toEqual([1, 2, 3, 4, 5, 6])
  })
})

describe('backfillOlderTranscriptPage', () => {
  beforeEach(() => {
    $transcriptTailBySessionId.set({})
    _resetTranscriptBackfillForTests()
    vi.mocked(getOlderSessionMessages).mockReset()
    vi.mocked(getSessionMessages).mockReset()
    vi.mocked(getSessionMessages).mockResolvedValue({ messages: [], session_id: 'stored-1' } as never)
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  const truncatedTail = (nextOffset = 120) => {
    recordTranscriptTail('stored-1', {
      messages: Array.from({ length: 120 }, (_, index) => row(index + nextOffset, `tail${index}`)),
      pagination: { limit: 120, offset: 0, order: 'latest', returned: 120 }
    })
  }

  it('fetches the recorded next offset and applies the converted page', async () => {
    truncatedTail()
    vi.mocked(getOlderSessionMessages).mockResolvedValue({
      messages: [row(1, 'older-1'), row(2, 'older-2')],
      pagination: { limit: 120, offset: 120, order: 'latest', returned: 2 },
      session_id: 'stored-1'
    } as never)

    const applyOlderPage = vi.fn()

    const applied = await backfillOlderTranscriptPage({
      storedSessionId: 'stored-1',
      isCurrent: () => true,
      applyOlderPage
    })

    expect(applied).toBe(true)
    expect(getOlderSessionMessages).toHaveBeenCalledWith('stored-1', undefined, 120)
    // The exhausted tail also fetches the origin page (see the #96875 tests
    // below), so assert the older page landed as the FIRST apply.
    expect(applyOlderPage.mock.calls[0][0].map((m: ChatMessage) => m.rowId)).toEqual([1, 2])
    // A short older page means the transcript is now fully loaded.
    expect(transcriptBackfillAvailable('stored-1')).toBe(false)
  })

  it('keeps the live tail in order when the fetched offset page includes subsequently persisted rows', async () => {
    recordTranscriptTail('stored-1', {
      messages: [row(4, 'four'), row(5, 'five'), row(6, 'six')],
      pagination: { limit: 3, offset: 0, order: 'latest', returned: 3 }
    })
    // Four rows persisted since hydration. Offset 3 now selects rows 5–7,
    // rather than a page wholly before the cached 4–6 tail.
    vi.mocked(getOlderSessionMessages).mockResolvedValue({
      messages: [row(5, 'five'), row(6, 'six'), row(7, 'seven')],
      pagination: { limit: 3, offset: 3, order: 'latest', returned: 3 },
      session_id: 'stored-1'
    } as never)

    let visible = [chat('four', 4), chat('five', 5), chat('six', 6)]

    const applied = await backfillOlderTranscriptPage({
      storedSessionId: 'stored-1',
      isCurrent: () => true,
      applyOlderPage: page => {
        visible = mergeOlderTranscriptPage(visible, page)
      }
    })

    expect(applied).toBe(true)
    expect(getOlderSessionMessages).toHaveBeenCalledWith('stored-1', undefined, 3)
    expect(visible.map(message => message.rowId)).toEqual([4, 5, 6, 7])
    expect(transcriptTailState('stored-1')?.nextOffset).toBe(6)

    vi.mocked(getOlderSessionMessages).mockResolvedValue({
      messages: [row(2, 'two'), row(3, 'three'), row(4, 'four')],
      pagination: { limit: 3, offset: 6, order: 'latest', returned: 3 },
      session_id: 'stored-1'
    } as never)

    await backfillOlderTranscriptPage({
      storedSessionId: 'stored-1',
      isCurrent: () => true,
      applyOlderPage: page => {
        visible = mergeOlderTranscriptPage(visible, page)
      }
    })

    expect(getOlderSessionMessages).toHaveBeenLastCalledWith('stored-1', undefined, 6)
    expect(visible.map(message => message.rowId)).toEqual([2, 3, 4, 5, 6, 7])
  })

  it('backfills the matching connection when two owners share one session id', async () => {
    const sourceA = { connectionId: 'source-a', profile: 'backend-a' }
    const sourceB = { connectionId: 'source-b', profile: 'backend-b' }

    const page = {
      messages: Array.from({ length: 120 }, (_, index) => row(index, `tail${index}`)),
      pagination: { limit: 120, offset: 0, order: 'latest' as const, returned: 120 }
    }

    recordTranscriptTail('same-session', page, sourceA)
    recordTranscriptTail('same-session', page, sourceB)
    vi.mocked(getOlderSessionMessages).mockResolvedValue({
      messages: [row(1, 'older')],
      pagination: { limit: 120, offset: 120, order: 'latest', returned: 1 },
      session_id: 'same-session'
    } as never)

    await backfillOlderTranscriptPage({
      storedSessionId: 'same-session',
      profile: sourceB,
      isCurrent: () => true,
      applyOlderPage: vi.fn()
    })

    expect(getOlderSessionMessages).toHaveBeenCalledWith('same-session', sourceB, 120)
    expect(transcriptTailState('same-session', sourceA)).toMatchObject({ possiblyTruncated: true })
    expect(transcriptTailState('same-session', sourceB)).toMatchObject({ possiblyTruncated: false })
    expect(transcriptTailState('same-session')).toBeUndefined()
  })

  it('keeps backfill available while pages keep coming back full', async () => {
    truncatedTail()
    vi.mocked(getOlderSessionMessages).mockResolvedValue({
      messages: Array.from({ length: 120 }, (_, index) => row(index, `older${index}`)),
      pagination: { limit: 120, offset: 120, order: 'latest', returned: 120 },
      session_id: 'stored-1'
    } as never)

    await backfillOlderTranscriptPage({ storedSessionId: 'stored-1', isCurrent: () => true, applyOlderPage: vi.fn() })

    expect(transcriptTailState('stored-1')).toMatchObject({ nextOffset: 240, possiblyTruncated: true })
  })

  it('falls back to the full transcript when a legacy backend returns no pagination metadata', async () => {
    truncatedTail()
    // Legacy backend: ignores limit/offset/order and one-shots everything.
    vi.mocked(getOlderSessionMessages).mockResolvedValue({
      messages: Array.from({ length: 700 }, (_, index) => row(index, `full${index}`)),
      session_id: 'stored-1'
    } as never)

    const applyOlderPage = vi.fn()

    const applied = await backfillOlderTranscriptPage({
      storedSessionId: 'stored-1',
      isCurrent: () => true,
      applyOlderPage
    })

    expect(applied).toBe(true)
    expect(applyOlderPage.mock.calls[0][0]).toHaveLength(700)
    // One-shot full transcript: the REST action retires.
    expect(transcriptBackfillAvailable('stored-1')).toBe(false)
  })

  it('discards a stale response after a session switch', async () => {
    truncatedTail()
    vi.mocked(getOlderSessionMessages).mockResolvedValue({
      messages: [row(1, 'older-1')],
      pagination: { limit: 120, offset: 120, order: 'latest', returned: 1 },
      session_id: 'stored-1'
    } as never)

    const applyOlderPage = vi.fn()

    const applied = await backfillOlderTranscriptPage({
      storedSessionId: 'stored-1',
      // The user switched sessions while the page was in flight.
      isCurrent: () => false,
      applyOlderPage
    })

    expect(applied).toBe(false)
    expect(applyOlderPage).not.toHaveBeenCalled()
    // Bookkeeping untouched: the next visit re-records the tail anyway.
    expect(transcriptTailState('stored-1')).toMatchObject({ nextOffset: 120, possiblyTruncated: true })
  })

  it('discards an older page when the same session tail was replaced during the fetch', async () => {
    truncatedTail()
    let resolvePage!: (value: unknown) => void
    vi.mocked(getOlderSessionMessages).mockReturnValue(
      // SAFETY: this controlled promise resolves with the exact RPC response shape below.
      new Promise(resolve => {
        resolvePage = resolve
      }) as never
    )
    const applyOlderPage = vi.fn()

    const pending = backfillOlderTranscriptPage({
      storedSessionId: 'stored-1',
      isCurrent: () => true,
      applyOlderPage
    })

    // Rewind/revalidation replaced the display tail without changing the route.
    recordTranscriptTail('stored-1', {
      messages: [row(900, 'replacement')],
      pagination: { limit: 120, offset: 0, order: 'latest', returned: 1 }
    })
    resolvePage({
      messages: [row(1, 'stale older row')],
      pagination: { limit: 120, offset: 120, order: 'latest', returned: 1 },
      session_id: 'stored-1'
    })

    expect(await pending).toBe(false)
    expect(applyOlderPage).not.toHaveBeenCalled()
    expect(transcriptTailState('stored-1')).toMatchObject({ nextOffset: 1, possiblyTruncated: false })
  })

  it('shares one in-flight fetch per stored session', async () => {
    truncatedTail()

    let resolvePage: (value: unknown) => void = () => {}

    vi.mocked(getOlderSessionMessages).mockReturnValue(
      // SAFETY: this controlled promise resolves with the exact RPC response shape below.
      new Promise(resolve => {
        resolvePage = resolve
      }) as never
    )

    const first = backfillOlderTranscriptPage({
      storedSessionId: 'stored-1',
      isCurrent: () => true,
      applyOlderPage: vi.fn()
    })

    const second = backfillOlderTranscriptPage({
      storedSessionId: 'stored-1',
      isCurrent: () => true,
      applyOlderPage: vi.fn()
    })

    expect(second).toBe(first)
    expect(getOlderSessionMessages).toHaveBeenCalledTimes(1)

    resolvePage({
      messages: [row(1, 'older-1')],
      pagination: { limit: 120, offset: 120, order: 'latest', returned: 1 },
      session_id: 'stored-1'
    })

    await first
  })

  it('resolves false without fetching when the tail is not truncated', async () => {
    recordTranscriptTail('stored-1', {
      messages: [row(1, 'only')],
      pagination: { limit: 120, offset: 0, order: 'latest', returned: 1 }
    })

    const applied = await backfillOlderTranscriptPage({
      storedSessionId: 'stored-1',
      isCurrent: () => true,
      applyOlderPage: vi.fn()
    })

    expect(applied).toBe(false)
    expect(getOlderSessionMessages).not.toHaveBeenCalled()
  })

  it('survives a fetch failure and leaves the action retryable', async () => {
    truncatedTail()
    vi.mocked(getOlderSessionMessages).mockRejectedValue(new Error('network down'))

    const applied = await backfillOlderTranscriptPage({
      storedSessionId: 'stored-1',
      isCurrent: () => true,
      applyOlderPage: vi.fn()
    })

    expect(applied).toBe(false)
    expect(transcriptBackfillAvailable('stored-1')).toBe(true)
  })

  // #96875: paging can reach the true top while the opening USER turn is still
  // missing — compaction projection stamps the durable opening row
  // display_kind=hidden (hydration drops it) or the last `latest` page begins
  // at the first assistant row. Once the tail bookkeeping reports the session
  // fully loaded, the oldest display rows must be fetched once and the opening
  // user turn prepended so "Show earlier" lands on the greeting.
  it('fetches the oldest page once paging is exhausted and prepends the opening user turn', async () => {
    truncatedTail()
    // The last older page: full paging history ends at the first assistant row.
    vi.mocked(getOlderSessionMessages).mockResolvedValue({
      messages: [{ ...row(10, 'first-assistant'), role: 'assistant' as const }],
      pagination: { limit: 120, offset: 120, order: 'latest', returned: 1 },
      session_id: 'stored-1'
    } as never)
    // The oldest page (order=oldest, offset 0) holds the greeting, hidden.
    vi.mocked(getSessionMessages).mockResolvedValue({
      messages: [
        {
          id: 1,
          role: 'user' as const,
          content: 'opening greeting',
          display_kind: 'hidden',
          timestamp: 1
        },
        { ...row(10, 'first-assistant'), role: 'assistant' as const }
      ],
      pagination: { limit: 20, offset: 0, order: 'oldest', returned: 2 },
      session_id: 'stored-1'
    } as never)

    const applyOlderPage = vi.fn()

    await backfillOlderTranscriptPage({
      storedSessionId: 'stored-1',
      isCurrent: () => true,
      applyOlderPage
    })

    expect(getSessionMessages).toHaveBeenCalledWith('stored-1', undefined, {
      includeCompacted: true,
      limit: 20,
      offset: 0,
      order: 'oldest'
    })
    // First call: the regular older page. Second call: the origin page.
    expect(applyOlderPage).toHaveBeenCalledTimes(2)
    const originPage = applyOlderPage.mock.calls[1][0] as ChatMessage[]
    expect(originPage[0].role).toBe('user')
    expect(originPage[0].parts[0]).toMatchObject({ type: 'text', text: 'opening greeting' })
  })

  it('does not fetch the oldest page while older pages remain', async () => {
    truncatedTail()
    vi.mocked(getOlderSessionMessages).mockResolvedValue({
      messages: Array.from({ length: 120 }, (_, index) => row(index, `older${index}`)),
      pagination: { limit: 120, offset: 120, order: 'latest', returned: 120 },
      session_id: 'stored-1'
    } as never)

    await backfillOlderTranscriptPage({
      storedSessionId: 'stored-1',
      isCurrent: () => true,
      applyOlderPage: vi.fn()
    })

    expect(getSessionMessages).not.toHaveBeenCalled()
  })

  it('skips the origin prepend when the user switched sessions mid-flight', async () => {
    truncatedTail()
    vi.mocked(getOlderSessionMessages).mockResolvedValue({
      messages: [row(10, 'first-assistant')],
      pagination: { limit: 120, offset: 120, order: 'latest', returned: 1 },
      session_id: 'stored-1'
    } as never)

    const applyOlderPage = vi.fn()

    await backfillOlderTranscriptPage({
      storedSessionId: 'stored-1',
      isCurrent: () => false,
      applyOlderPage
    })

    // The stale-response guard discards before any apply; no origin fetch ran.
    expect(applyOlderPage).not.toHaveBeenCalled()
  })

  it('keeps the already-applied older page when the origin fetch fails', async () => {
    truncatedTail()
    vi.mocked(getOlderSessionMessages).mockResolvedValue({
      messages: [row(10, 'first-assistant')],
      pagination: { limit: 120, offset: 120, order: 'latest', returned: 1 },
      session_id: 'stored-1'
    } as never)
    vi.mocked(getSessionMessages).mockRejectedValue(new Error('network down'))

    const applyOlderPage = vi.fn()

    const applied = await backfillOlderTranscriptPage({
      storedSessionId: 'stored-1',
      isCurrent: () => true,
      applyOlderPage
    })

    expect(applied).toBe(true)
    expect(applyOlderPage).toHaveBeenCalledTimes(1)
  })
})

describe('unhideOpeningUserRows', () => {
  it('clears hidden on the opening user row so hydration keeps the greeting', () => {
    const messages = unhideOpeningUserRows([
      {
        id: 1,
        role: 'user',
        content: 'hello there',
        display_kind: 'hidden',
        timestamp: 1
      }
    ])

    expect(messages[0].display_kind).toBeUndefined()
    expect(messages[0].display_content).toBe('hello there')
  })

  it('leaves a visible opening user row untouched', () => {
    const messages = unhideOpeningUserRows([row(1, 'plain greeting')])

    expect(messages[0].display_kind).toBeUndefined()
    expect(messages[0].display_content).toBeUndefined()
  })

  it('leaves hidden rows after the opening user turn alone', () => {
    const messages = unhideOpeningUserRows([
      row(1, 'greeting'),
      { id: 2, role: 'user', content: 'scaffolding', display_kind: 'hidden', timestamp: 2 }
    ])

    expect(messages[1].display_kind).toBe('hidden')
  })

  it('leaves a hidden opening row without content hidden', () => {
    const messages = unhideOpeningUserRows([{ id: 1, role: 'user', content: '', display_kind: 'hidden', timestamp: 1 }])

    expect(messages[0].display_kind).toBe('hidden')
  })
})
