import { describe, expect, it } from 'vitest'

import { graftRefreshedTailOntoBackfill } from '@/app/chat/transcript-backfill'
import {
  type ChatMessage,
  type ChatMessagePart,
  stripPendingClarifyProjectionForCache,
  toChatMessages
} from '@/lib/chat-messages'
import type { SessionMessage } from '@/types/hermes'

const row = (rowId: number, role: ChatMessage['role']): ChatMessage => ({
  id: `stored-${rowId}`,
  rowId,
  role,
  parts: [{ type: 'text', text: `Answer or prompt ${rowId}` }]
})

const question = (toolName: string, result: unknown): ChatMessage => ({
  id: `synthetic-${toolName}`,
  role: 'assistant',
  pending: false,
  parts: [
    {
      type: 'tool-call',
      toolCallId: `request-${toolName}`,
      toolName,
      args: { questions: [{ question: 'Switch to:', choices: ['On', 'Off'] }] },
      ...(result !== undefined ? { result } : { completedAt: 31 })
    }
  ]
})

const history = () => ({
  previous: [row(10, 'user'), row(30, 'assistant')],
  // An overlap with a previously unheld row takes the stored-id merge path.
  fresh: [row(20, 'assistant'), row(30, 'assistant'), row(40, 'user'), row(50, 'assistant')]
})

describe('settled tool rows in an overlapping transcript refresh', () => {
  it('keeps historical cards beside their durable anchor without losing results or backend rows', () => {
    const results = [
      { outcome: 'cancelled', responses: [{ question: 'Switch to:', status: 'unanswered', user_response: null }] },
      { responses: [{ question: 'Switch to:', status: 'answered', user_response: 'On' }] },
      undefined, // A lost completion sealed at turn settle is also history.
      null,
      false,
      0,
      ''
    ]

    for (const result of results) {
      const { previous, fresh } = history()
      // The connection-request sibling uses the same synthetic tool-only shape.
      const cards = ['clarify', 'setup_mcp'].map(name => question(name, result))
      const window = [...previous, ...cards]
      const cached = stripPendingClarifyProjectionForCache(window)
      const historicalCards = cached.slice(previous.length)
      expect(historicalCards).toEqual(cards)

      if (result !== undefined) {
        expect(cached).toBe(window)
      }

      // A separate durable occurrence must survive even with identical question text.
      const backendCard = { ...question('clarify', result), id: 'stored-question', rowId: 20 }
      fresh[0] = backendCard

      const merged = graftRefreshedTailOntoBackfill(fresh, cached)
      const expected = [previous[0], ...fresh.slice(0, 2), ...historicalCards, ...fresh.slice(2)]

      expect(merged).toEqual(expected)
      expected.forEach((message, index) => expect(merged[index]).toBe(message))
      // Another overlapping page can fill a gap after the original anchor.
      // Historical cards must not start travelling with the next durable row.
      const overlapping = [row(15, 'assistant'), ...fresh.slice(0, 2), row(35, 'assistant'), ...fresh.slice(2)]
      const repeated = graftRefreshedTailOntoBackfill(overlapping, merged)
      expect(repeated).toEqual([previous[0], ...overlapping.slice(0, 3), ...historicalCards, ...overlapping.slice(3)])
      expect(cached).toEqual([...previous, ...historicalCards])
    }
  })

  it('leaves active or unbound content at the live tail and preserves authoritative no-op results', () => {
    const { previous, fresh } = history()
    const settled = question('clarify', { response: 'On' })

    const open = {
      ...settled,
      parts: settled.parts.map(part => {
        if (part.type !== 'tool-call') {
          return part
        }

        const { completedAt: _at, result: _result, ...unsettled } = part

        return unsettled
      })
    }

    const text: ChatMessagePart = { type: 'text', text: 'An answer not yet bound to history' }

    const candidates: ChatMessage[] = [
      { ...settled, pending: true },
      { ...settled, interim: true },
      { ...settled, error: 'Incomplete turn' },
      { ...settled, durableComplete: false },
      { ...settled, persistedTurn: { complete: false, row_ids: [] } },
      open,
      { ...settled, parts: [text] },
      { ...settled, parts: [...settled.parts, text] },
      { ...settled, parts: [...settled.parts, ...open.parts] },
      { ...settled, parts: settled.parts.map(part => ({ ...part, sourceRowId: 60 })) },
      { ...settled, role: 'user' },
      { ...settled, parts: [] }
    ]

    for (const candidate of candidates) {
      const merged = graftRefreshedTailOntoBackfill(fresh, [...previous, candidate])
      expect(merged).toEqual([previous[0], ...fresh, candidate])
      expect(merged.at(-1)).toBe(candidate)
      // A settled tool after unbound content cannot borrow an earlier turn's anchor.
      expect(graftRefreshedTailOntoBackfill(fresh, [...previous, candidate, settled])).toEqual([
        previous[0],
        ...fresh,
        candidate,
        settled
      ])
    }

    // A shared page-local tool fold remains an ordering barrier even when
    // fresh-page precedence suppresses its old copy in the window.
    const localRows = toChatMessages([
      { id: 10, role: 'user', content: 'Earlier prompt', timestamp: 1 },
      { id: 30, role: 'user', content: 'Read a file', timestamp: 1 },
      { id: 32, role: 'tool', content: 'File read complete', timestamp: 1 }
    ] as SessionMessage[])

    const pageRows = toChatMessages([
      { id: 20, role: 'assistant', content: 'Earlier reply', timestamp: 1 },
      { id: 30, role: 'user', content: 'Read a file', timestamp: 1 },
      { id: 32, role: 'tool', content: 'File read complete', timestamp: 1 },
      { id: 40, role: 'user', content: 'Next prompt', timestamp: 1 },
      { id: 50, role: 'assistant', content: 'Next reply', timestamp: 1 }
    ] as SessionMessage[])

    expect(localRows.at(-1)?.rowId).toBeUndefined()
    expect(localRows.at(-1)?.durableComplete).toBe(false)
    expect(graftRefreshedTailOntoBackfill(pageRows, [...localRows, settled])).toEqual([
      localRows[0],
      ...pageRows,
      settled
    ])

    const unanchored = [settled, ...previous]
    expect(graftRefreshedTailOntoBackfill(fresh, unanchored)).toEqual([settled, previous[0], ...fresh])
    expect(graftRefreshedTailOntoBackfill(fresh, [])).toBe(fresh)
    expect(graftRefreshedTailOntoBackfill(fresh, fresh)).toBe(fresh)
    expect(graftRefreshedTailOntoBackfill(fresh, [row(100, 'user'), settled])).toBe(fresh)
    const authoritative = { ...settled, pending: true }
    expect(graftRefreshedTailOntoBackfill([...fresh, authoritative], [...previous, settled])).toEqual([
      previous[0],
      ...fresh,
      authoritative
    ])
  })
})
