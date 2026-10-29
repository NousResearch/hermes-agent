import type { PersistedTurn } from '@hermes/shared'
import { describe, expect, it } from 'vitest'

import { graftRefreshedTailOntoBackfill } from '@/app/chat/transcript-backfill'
import { bindStoppedTurnReceipt } from '@/app/session/hooks/use-message-stream/interrupted-reply'
import { type ChatMessage, chatMessageText, toChatMessages } from '@/lib/chat-messages'
import type { SessionMessage } from '@/types/hermes'

/**
 * A long chat's newest page begins inside an older tool turn, so the refresh
 * graft pairs rows by stored id. The turn that just ran was drawn live and
 * carries stored ids on some rows at most. The page holds every stored row of
 * that turn, and the turn's receipt says so: each of its messages must render
 * once (#139741 whole turn twice, #141771 narration again under the answer).
 */

const at = (id: number) => 1_000 + id

const user = (id: number, text: string): SessionMessage =>
  ({ id, role: 'user', content: text, timestamp: at(id) }) as SessionMessage

const narration = (id: number, text: string, call: string): SessionMessage =>
  ({
    id,
    role: 'assistant',
    content: text,
    timestamp: at(id),
    tool_calls: [{ id: call, type: 'function', function: { name: 'terminal', arguments: '{"command":"echo"}' } }]
  }) as SessionMessage

const tool = (id: number, call: string): SessionMessage =>
  ({ id, role: 'tool', content: 'ok', tool_call_id: call, tool_name: 'terminal', timestamp: at(id) }) as SessionMessage

const answer = (id: number, text: string): SessionMessage =>
  ({ id, role: 'assistant', content: text, timestamp: at(id) }) as SessionMessage

/** An older turn of three narrated tool rounds: rows first..first+7. */
const olderTurn = (first: number, tag: string): SessionMessage[] => [
  user(first, `${tag} prompt`),
  narration(first + 1, `${tag} look one`, `${tag}-1`),
  tool(first + 2, `${tag}-1`),
  narration(first + 3, `${tag} look two`, `${tag}-2`),
  tool(first + 4, `${tag}-2`),
  narration(first + 5, `${tag} look three`, `${tag}-3`),
  tool(first + 6, `${tag}-3`),
  answer(first + 7, `${tag} answer`)
]

const texts = (messages: ChatMessage[]) => messages.map(message => chatMessageText(message)).join('\n')

const count = (haystack: string, needle: string) => haystack.split(needle).length - 1

const live = (id: string, role: ChatMessage['role'], text: string, extra: Partial<ChatMessage> = {}): ChatMessage => ({
  id,
  role,
  parts: [{ type: 'text', text }],
  ...extra
})

describe('a refreshed page that holds the live turn whole', () => {
  // Window: two older turns hydrated from an earlier page. Page: starts at the
  // second narration of the first older turn (its fold is addressed by a row
  // the window does not use as a message id), then the live turn's stored rows.
  const windowRows = [...olderTurn(1, 'old-a'), ...olderTurn(9, 'old-b')]
  const liveRows = [user(17, 'steered prompt'), narration(18, 'live look', 'live-1'), tool(19, 'live-1')]

  it('draws a steered turn once when its receipt is not complete', () => {
    const receipt: PersistedTurn = {
      row_ids: [17, 18, 19, 20, 21],
      complete: false,
      user_row_ids: [17, 20],
      submission_id: 'sub-steer'
    }

    const window = [
      ...toChatMessages(windowRows),
      live('user-sub-steer', 'user', 'steered prompt', { rowId: 17 }),
      live('assistant-stream-1', 'assistant', 'live look', { interim: false }),
      live('user-other', 'user', 'change course', { rowId: 20 }),
      live('assistant-stream-2', 'assistant', 'steered answer', { rowId: 21, persistedTurn: receipt })
    ]

    const page = toChatMessages([
      ...olderTurn(1, 'old-a').slice(3),
      ...olderTurn(9, 'old-b'),
      ...liveRows,
      user(20, 'change course'),
      answer(21, 'steered answer')
    ])

    const merged = texts(graftRefreshedTailOntoBackfill(page, window))

    for (const text of ['steered prompt', 'live look', 'change course', 'steered answer', 'old-b look two']) {
      expect(count(merged, text), text).toBe(1)
    }
  })

  it('keeps an answer the receipt does not cover when the turn ended without storing it', () => {
    // The final answer failed to store: the partial receipt addresses only rows 17-19, and a later
    // stored turn follows, so the page reaches past the receipt without carrying the answer.
    const receipt: PersistedTurn = {
      row_ids: [17, 18, 19],
      complete: false,
      user_row_id: 17,
      submission_id: 'sub-lost'
    }

    const window = [
      ...toChatMessages(windowRows),
      live('user-sub-lost', 'user', 'steered prompt', { rowId: 17 }),
      live('assistant-stream-1', 'assistant', 'live look', { interim: false }),
      live('assistant-stream-2', 'assistant', 'unstored answer', { persistedTurn: receipt })
    ]

    const page = toChatMessages([
      ...olderTurn(1, 'old-a').slice(3),
      ...olderTurn(9, 'old-b'),
      ...liveRows,
      user(20, 'later prompt'),
      answer(21, 'later answer')
    ])

    expect(count(texts(graftRefreshedTailOntoBackfill(page, window)), 'unstored answer')).toBe(1)
  })

  it('draws an older turn the page boundary cuts once', () => {
    const window = [...toChatMessages(windowRows), live('user-x', 'user', 'next prompt', { rowId: 17 })]

    // The page starts at the first turn's second narration row: its fold of
    // that turn is addressed by a row the window's fold carries inside it.
    const page = toChatMessages([...olderTurn(1, 'old-a').slice(3), ...olderTurn(9, 'old-b'), user(17, 'next prompt')])
    const merged = texts(graftRefreshedTailOntoBackfill(page, window))

    for (const text of ['old-a look one', 'old-a look two', 'old-a look three', 'old-a answer', 'next prompt']) {
      expect(count(merged, text), text).toBe(1)
    }
  })

  it('draws a stopped turn once after its interrupted completion binds the receipt', () => {
    const receipt: PersistedTurn = {
      row_ids: [17, 18, 19, 20],
      complete: false,
      user_row_id: 17,
      user_row_ids: [17],
      submission_id: 'sub-stop'
    }

    const stopped = bindStoppedTurnReceipt(
      [
        ...toChatMessages(windowRows),
        live('user-sub-stop', 'user', 'steered prompt'),
        live('assistant-stream-1', 'assistant', 'live look partial')
      ],
      receipt
    )

    const page = toChatMessages([
      ...olderTurn(1, 'old-a').slice(3),
      ...olderTurn(9, 'old-b'),
      ...liveRows,
      answer(20, 'partial')
    ])

    const merged = texts(graftRefreshedTailOntoBackfill(page, stopped))

    expect(count(merged, 'steered prompt')).toBe(1)
    expect(count(merged, 'live look')).toBe(1)
  })

  it('draws a stopped turn once when a page merged before its interrupted completion', () => {
    const receipt: PersistedTurn = {
      row_ids: [17, 18, 19, 20],
      complete: false,
      user_row_id: 17,
      user_row_ids: [17],
      submission_id: 'sub-race'
    }

    const page = toChatMessages([
      ...olderTurn(1, 'old-a').slice(3),
      ...olderTurn(9, 'old-b'),
      ...liveRows,
      answer(20, 'partial')
    ])

    // Stop settled the turn on screen, a refresh merged the page (which holds
    // the turn's stored rows) while the live rows had no stored id, and only
    // then did the interrupted completion bring the receipt.
    const refreshed = graftRefreshedTailOntoBackfill(page, [
      ...toChatMessages(windowRows),
      live('user-sub-race', 'user', 'steered prompt'),
      live('assistant-stream-1', 'assistant', 'live look partial')
    ])

    const stopped = texts(bindStoppedTurnReceipt(refreshed, receipt))

    expect(count(stopped, 'steered prompt')).toBe(1)
    expect(count(stopped, 'live look')).toBe(1)
    expect(count(stopped, 'old-b answer')).toBe(1)
  })

  it('draws a turn stopped during its tool run once, keeping its stopped mark', () => {
    // No reply text: the turn ends on its tool result and an empty assistant
    // row, neither of which a page paints.
    const receipt: PersistedTurn = {
      row_ids: [17, 18, 19, 20],
      complete: false,
      user_row_id: 17,
      user_row_ids: [17],
      submission_id: 'sub-tool-stop'
    }

    const stopped = bindStoppedTurnReceipt(
      [
        ...toChatMessages(windowRows),
        live('user-sub-tool-stop', 'user', 'steered prompt'),
        {
          ...live('assistant-stream-1', 'assistant', 'live look'),
          parts: [
            { type: 'text', text: 'live look' },
            {
              type: 'tool-call',
              toolCallId: 'live-1',
              toolName: 'terminal',
              args: { command: 'echo' },
              argsText: '{"command":"echo"}',
              interrupted: true
            }
          ],
          interrupted: true,
          pending: false
        }
      ],
      receipt
    )

    const page = toChatMessages([
      ...olderTurn(1, 'old-a').slice(3),
      ...olderTurn(9, 'old-b'),
      ...liveRows,
      answer(20, '')
    ])

    const merged = graftRefreshedTailOntoBackfill(page, stopped)

    expect(count(texts(merged), 'steered prompt')).toBe(1)
    expect(count(texts(merged), 'live look')).toBe(1)
    expect(merged.find(message => chatMessageText(message).includes('live look'))?.interrupted).toBe(true)
  })

  it('keeps a live turn the page does not reach yet', () => {
    const receipt: PersistedTurn = { row_ids: [17, 18, 19, 20, 21], complete: false, submission_id: 'sub-late' }

    const window = [
      ...toChatMessages(windowRows),
      live('user-sub-late', 'user', 'late prompt'),
      live('assistant-stream-1', 'assistant', 'late answer', { persistedTurn: receipt })
    ]

    // A stale page that ends before the turn's last row.
    const page = toChatMessages([...olderTurn(1, 'old-a').slice(3), ...olderTurn(9, 'old-b')])
    const merged = texts(graftRefreshedTailOntoBackfill(page, window))

    expect(count(merged, 'late prompt')).toBe(1)
    expect(count(merged, 'late answer')).toBe(1)
  })
})
