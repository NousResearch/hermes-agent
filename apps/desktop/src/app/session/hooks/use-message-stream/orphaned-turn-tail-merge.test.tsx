import type { GatewayEventName, PersistedTurn } from '@hermes/shared'
import { act, cleanup } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { graftRefreshedTailOntoBackfill } from '@/app/chat/transcript-backfill'
import {
  type ChatMessage,
  chatMessageText,
  preserveLocalAssistantErrors,
  preserveLocalSystemNotices,
  toChatMessages
} from '@/lib/chat-messages'
import type { SessionMessage } from '@/types/hermes'

import { preserveLocalPendingTurnMessages } from '../use-session-actions/utils'

import { renderMessageStream } from './test-harness'

const SID = 'orphaned-turn'

afterEach(cleanup)

const toolCall = (id: string) => [
  { id, type: 'function', function: { name: 'terminal', arguments: '{"command":"true"}' } }
]

// An older tool turn (rows 1-8), turn N (9-12) and turn N+1 (13-16).
const ROWS: SessionMessage[] = [
  { id: 1, role: 'user', content: 'Old prompt', timestamp: 1 },
  { id: 2, role: 'assistant', content: 'Old narration', tool_calls: toolCall('t1'), timestamp: 2 },
  { id: 3, role: 'tool', content: 'ok', tool_call_id: 't1', tool_name: 'terminal', timestamp: 3 },
  { id: 4, role: 'assistant', content: '', tool_calls: toolCall('t2'), timestamp: 4 },
  { id: 5, role: 'tool', content: 'ok', tool_call_id: 't2', tool_name: 'terminal', timestamp: 5 },
  { id: 6, role: 'assistant', content: 'Old mid narration', tool_calls: toolCall('t3'), timestamp: 6 },
  { id: 7, role: 'tool', content: 'ok', tool_call_id: 't3', tool_name: 'terminal', timestamp: 7 },
  { id: 8, role: 'assistant', content: 'Old final', timestamp: 8 },
  { id: 9, role: 'user', content: 'Turn N prompt', timestamp: 9 },
  { id: 10, role: 'assistant', content: 'N narration', tool_calls: toolCall('n1'), timestamp: 10 },
  { id: 11, role: 'tool', content: 'ok', tool_call_id: 'n1', tool_name: 'terminal', timestamp: 11 },
  { id: 12, role: 'assistant', content: 'FINAL A of turn N', timestamp: 12 },
  { id: 13, role: 'user', content: 'Turn N+1 prompt', timestamp: 13 },
  { id: 14, role: 'assistant', content: '', tool_calls: toolCall('m1'), timestamp: 14 },
  { id: 15, role: 'tool', content: 'ok', tool_call_id: 'm1', tool_name: 'terminal', timestamp: 15 },
  { id: 16, role: 'assistant', content: 'FINAL B of turn N+1', timestamp: 16 }
]

const userBubble = (id: string, text: string, rowId: number): ChatMessage => ({
  id,
  role: 'user',
  parts: [{ type: 'text', text }],
  rowId
})

const RECEIPT_N = {
  row_ids: [9, 10, 11, 12],
  complete: true,
  user_row_id: 9,
  user_row_ids: [9],
  final_assistant_row_id: 12
}

const RECEIPT_N1: PersistedTurn = {
  row_ids: [13, 14, 15, 16],
  complete: true,
  user_row_id: 13,
  user_row_ids: [13],
  final_assistant_row_id: 16
}

/** Streams turn N then turn N+1 live, returning the window as the client holds it. */
async function streamTwoTurns(receiptN: PersistedTurn | undefined): Promise<ChatMessage[]> {
  const h = renderMessageStream(SID)

  const send = (type: GatewayEventName, payload: Record<string, unknown> = {}) =>
    act(() => h.handleEvent({ type, payload, session_id: SID }))

  const push = (message: ChatMessage) => h.states.set(SID, { ...h.state(), messages: [...h.state().messages, message] })

  h.states.set(SID, { ...h.state(), messages: toChatMessages(ROWS.slice(0, 8)) })

  // Turn N: no refresh follows it before the next prompt.
  push(userBubble('user-a', 'Turn N prompt', 9))
  await send('message.start', { turn_id: 'tn' })
  await send('message.delta', { text: 'N narration', turn_id: 'tn', segment: 0, offset: 0 })
  await send('message.interim', { text: 'N narration', already_streamed: true, turn_id: 'tn', segment: 0 })
  await send('tool.start', { tool_id: 'n1', name: 'terminal', args: { command: 'true' } })
  await send('tool.complete', { tool_id: 'n1', name: 'terminal', result: 'ok' })
  await send('message.delta', { text: 'FINAL A of turn N', turn_id: 'tn', segment: 1, offset: 0 })

  await send('message.complete', {
    text: 'FINAL A of turn N',
    status: 'complete',
    ...(receiptN ? { persisted_turn: receiptN } : {})
  })

  push(userBubble('user-b', 'Turn N+1 prompt', 13))
  await send('message.start', { turn_id: 'tn1' })
  await send('tool.start', { tool_id: 'm1', name: 'terminal', args: { command: 'true' } })
  await send('tool.complete', { tool_id: 'm1', name: 'terminal', result: 'ok' })
  await send('message.delta', { text: 'FINAL B of turn N+1', turn_id: 'tn1', segment: 0, offset: 0 })
  await send('message.complete', { text: 'FINAL B of turn N+1', status: 'complete', persisted_turn: RECEIPT_N1 })

  return h.state().messages
}

/** The post-turn refresh chain (use-background-sync reconcile) over the newest page. */
const refresh = (current: ChatMessage[], page: SessionMessage[]): ChatMessage[] =>
  preserveLocalSystemNotices(
    preserveLocalAssistantErrors(
      preserveLocalPendingTurnMessages(graftRefreshedTailOntoBackfill(toChatMessages(page), current), current),
      current
    ),
    current
  )

const count = (messages: ChatMessage[], needle: string) =>
  messages.filter(m => m.role === 'assistant' && chatMessageText(m).includes(needle)).length

const order = (messages: ChatMessage[], needles: string[]) =>
  needles.map(needle => messages.findIndex(m => chatMessageText(m).includes(needle)))

describe('refresh after a turn whose live reply never bound to its stored row', () => {
  // The newest page (rows 4..16) starts inside the older tool turn, so its first
  // fold has no counterpart in the window and the overlap merge takes over.
  const page = ROWS.slice(3)

  it.each([
    ['no persisted_turn', undefined],
    ['a receipt without user row ids', { row_ids: [9, 10, 11, 12], complete: false, final_assistant_row_id: 12 }],
    ['a complete receipt', RECEIPT_N]
  ] as [string, PersistedTurn | undefined][])(
    'paints turn N once, before turn N+1, with %s',
    async (_label, receipt) => {
      const merged = refresh(await streamTwoTurns(receipt), page)

      expect(count(merged, 'FINAL A of turn N')).toBe(1)
      expect(count(merged, 'FINAL B of turn N+1')).toBe(1)

      const [a, next, b] = order(merged, ['FINAL A of turn N', 'Turn N+1 prompt', 'FINAL B of turn N+1'])

      expect(a).toBeGreaterThanOrEqual(0)
      expect(a).toBeLessThan(next)
      expect(next).toBeLessThan(b)
      expect(merged.at(-1)?.role).toBe('assistant')
      expect(chatMessageText(merged.at(-1)!)).toBe('FINAL B of turn N+1')
    }
  )
})
