import type { PersistedTurn } from '@hermes/shared'
import { describe, expect, it } from 'vitest'

import { graftRefreshedTailOntoBackfill } from '@/app/chat/transcript-backfill'
import { preserveLocalPendingTurnMessages } from '@/app/session/hooks/use-session-actions/utils'
import {
  type ChatMessage,
  chatMessageText,
  preserveLocalAssistantErrors,
  preserveLocalSystemNotices,
  toChatMessages
} from '@/lib/chat-messages'
import type { SessionMessage } from '@/types/hermes'

/**
 * Rows the renderer draws and never stores (a /btw echo and its answer, a
 * fallback-switch notice) must survive the refresh that settles the turn they
 * landed in. A steer puts later rows of the same turn below them.
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

const processComplete = (id: number): SessionMessage =>
  ({
    id,
    role: 'user',
    content: 'Background Process Finished',
    timestamp: at(id),
    display_kind: 'process_complete'
  }) as SessionMessage

const olderTurn = (first: number, tag: string): SessionMessage[] => [
  user(first, `${tag} prompt`),
  narration(first + 1, `${tag} look one`, `${tag}-1`),
  tool(first + 2, `${tag}-1`),
  narration(first + 3, `${tag} look two`, `${tag}-2`),
  tool(first + 4, `${tag}-2`),
  answer(first + 5, `${tag} answer`)
]

const live = (id: string, role: ChatMessage['role'], text: string, extra: Partial<ChatMessage> = {}): ChatMessage => ({
  id,
  role,
  parts: [{ type: 'text', text }],
  ...extra
})

const count = (messages: ChatMessage[], needle: string) =>
  messages
    .map(message => chatMessageText(message))
    .join('\n')
    .split(needle).length - 1

/** The refresh composition the post-turn and sessions.changed reads apply. */
const refresh = (page: ChatMessage[], window: ChatMessage[]) =>
  preserveLocalSystemNotices(
    preserveLocalAssistantErrors(
      preserveLocalPendingTurnMessages(graftRefreshedTailOntoBackfill(page, window), window),
      window
    ),
    window
  )

describe('local notices inside a steered turn', () => {
  const receipt: PersistedTurn = {
    row_ids: [13, 14, 15, 16, 17],
    complete: false,
    user_row_ids: [13, 16],
    submission_id: 'sub-steer'
  }

  const notices = [
    live('system-1-btw', 'system', 'btw btw_1 — answering from a conversation snapshot'),
    live('btw-complete-btw_1', 'system', '[btw "side?" (btw_1)]\nSIDE ANSWER'),
    live('fallback-switch-1', 'system', 'Switched to fallback model')
  ]

  const turnRows = [user(13, 'steered prompt'), narration(14, 'live look', 'live-1'), tool(15, 'live-1')]

  it.each([
    // Long chat: the newest page starts inside an older turn (stored-id merge).
    ['a page that starts inside an older turn', [...olderTurn(1, 'old-a').slice(3), ...olderTurn(7, 'old-b')]],
    // Short chat: the page is the whole conversation (taken as is).
    ['a page that holds the whole chat', [...olderTurn(1, 'old-a'), ...olderTurn(7, 'old-b')]]
  ])('keep their place and the turn draws once after %s', (_label, pageHead) => {
    const window = [
      ...toChatMessages([...olderTurn(1, 'old-a'), ...olderTurn(7, 'old-b')]),
      live('user-sub-steer', 'user', 'steered prompt', { rowId: 13 }),
      live('assistant-stream-1', 'assistant', 'live look', { interim: false }),
      ...notices,
      live('user-other', 'user', 'change course', { rowId: 16 }),
      live('assistant-stream-2', 'assistant', 'steered answer', { rowId: 17, persistedTurn: receipt })
    ]

    const page = toChatMessages([...pageHead, ...turnRows, user(16, 'change course'), answer(17, 'steered answer')])
    const merged = refresh(page, window)

    for (const text of ['SIDE ANSWER', 'answering from a conversation snapshot', 'Switched to fallback model']) {
      expect(count(merged, text), text).toBe(1)
    }

    for (const text of ['steered prompt', 'live look', 'change course', 'steered answer', 'old-b look two']) {
      expect(count(merged, text), text).toBe(1)
    }

    // Arrival order: after the turn's prompt, before anything newer than the turn.
    const side = merged.findIndex(message => message.id === 'btw-complete-btw_1')
    expect(side).toBeGreaterThan(merged.findIndex(message => chatMessageText(message) === 'steered prompt'))
  })

  it('stay inside their own turn when a later turn whose prompt only the page holds follows it', () => {
    // The background process the turn started finished after it and ran its
    // own turn; the window drew that turn's reply but never its stored prompt.
    const auto: PersistedTurn = { row_ids: [18, 19], complete: true, user_row_ids: [18], submission_id: 'auto' }

    const window = [
      ...toChatMessages([...olderTurn(1, 'old-a'), ...olderTurn(7, 'old-b')]),
      live('user-sub-steer', 'user', 'steered prompt', { rowId: 13 }),
      live('assistant-stream-1', 'assistant', 'live look', { interim: false }),
      ...notices,
      live('user-other', 'user', 'change course', { rowId: 16 }),
      live('assistant-stream-2', 'assistant', 'steered answer', { rowId: 17, persistedTurn: receipt }),
      live('assistant-auto', 'assistant', 'auto reply', { rowId: 19, persistedTurn: auto })
    ]

    const page = toChatMessages([
      ...olderTurn(1, 'old-a').slice(3),
      ...olderTurn(7, 'old-b'),
      ...turnRows,
      user(16, 'change course'),
      answer(17, 'steered answer'),
      processComplete(18),
      answer(19, 'auto reply')
    ])

    const merged = refresh(page, window)
    const indexOf = (text: string) => merged.findIndex(message => chatMessageText(message).includes(text))

    for (const text of ['SIDE ANSWER', 'answering from a conversation snapshot', 'auto reply', 'change course']) {
      expect(count(merged, text), text).toBe(1)
    }

    // In front of the steer, as drawn live: never after the next turn's prompt.
    expect(indexOf('SIDE ANSWER')).toBeGreaterThan(indexOf('steered prompt'))
    expect(indexOf('SIDE ANSWER')).toBeLessThan(indexOf('change course'))
    expect(indexOf('Switched to fallback model')).toBeLessThan(indexOf('change course'))
  })

  it('stay in front of a steer prompt the window drew before its row was written', () => {
    const window = [
      ...toChatMessages([...olderTurn(1, 'old-a'), ...olderTurn(7, 'old-b')]),
      live('user-sub-steer', 'user', 'steered prompt', { rowId: 13 }),
      live('assistant-stream-1', 'assistant', 'live look', { interim: false }),
      ...notices,
      live('user-other', 'user', 'change course'),
      live('assistant-stream-2', 'assistant', 'steered answer', { rowId: 17, persistedTurn: receipt })
    ]

    const page = toChatMessages([
      ...olderTurn(1, 'old-a').slice(3),
      ...olderTurn(7, 'old-b'),
      ...turnRows,
      user(16, 'change course'),
      answer(17, 'steered answer')
    ])

    const merged = refresh(page, window)
    const indexOf = (text: string) => merged.findIndex(message => chatMessageText(message).includes(text))

    for (const text of ['SIDE ANSWER', 'change course', 'steered answer']) {
      expect(count(merged, text), text).toBe(1)
    }

    expect(indexOf('SIDE ANSWER')).toBeGreaterThan(indexOf('steered prompt'))
    expect(indexOf('SIDE ANSWER')).toBeLessThan(indexOf('change course'))
  })
})
