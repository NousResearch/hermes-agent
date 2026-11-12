import type { GatewayEventName } from '@hermes/shared'
import { act, cleanup } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { type ChatMessage, chatMessageText, toChatMessages } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'
import type { SessionMessage } from '@/types/hermes'

import { preserveLocalPendingTurnMessages } from '../use-session-actions/utils'

import { renderMessageStream } from './test-harness'

afterEach(cleanup)

const FULL = { row_ids: [1, 2, 3, 4, 5, 6], user_row_id: 1, final_assistant_row_id: 6, complete: true }

// #119540: the live stream seals each interim segment (with the tool calls it
// ran) as its own bubble; stored history folds the turn into one row. Without a
// complete persistence receipt (compaction mid-turn, older gateway) the post-turn
// refresh re-appended every segment after the first below the folded reply.
it.each([
  ['a complete receipt', FULL],
  ['a partial receipt', { ...FULL, complete: false }],
  ['no receipt', null]
])('a refreshed multi-segment tool turn renders each segment once with %s', async (_label, receipt) => {
  const sid = 'folded-segments'
  const states = new Map<string, ClientSessionState>()

  states.set(sid, {
    ...createClientSessionState(),
    messages: [{ id: 'user-1', role: 'user', rowId: 1, parts: [{ type: 'text', text: 'q' }] }]
  })

  const stream = renderMessageStream(sid, { states })

  const send = (type: GatewayEventName, payload: Record<string, unknown> = {}) =>
    act(() => stream.handleEvent({ type, payload, session_id: sid }))

  const history: SessionMessage[] = [{ id: 1, role: 'user', content: 'q' }]

  await send('message.start')

  for (const [index, comment] of ['First segment.', 'Second segment.'].entries()) {
    await send('message.delta', { text: comment })
    await send('message.interim', { text: comment, already_streamed: true })
    const id = `tool-${index}`
    await send('tool.start', { name: 'read_file', tool_id: id, args: {} })
    await send('tool.complete', { name: 'read_file', tool_id: id, result: 'read' })
    history.push(
      {
        id: index * 2 + 2,
        role: 'assistant',
        content: comment,
        tool_calls: [{ id, type: 'function', function: { name: 'read_file', arguments: '{}' } }]
      },
      { id: index * 2 + 3, role: 'tool', content: 'read', tool_call_id: id }
    )
  }

  await send('message.delta', { text: 'Done.' })
  await send('message.complete', { text: 'Done.', ...(receipt ? { persisted_turn: receipt } : {}) })
  history.push({ id: 6, role: 'assistant', content: 'Done.' })

  const durable = toChatMessages(history)
  const merged = preserveLocalPendingTurnMessages(durable, stream.state().messages as ChatMessage[])

  expect(merged.map(chatMessageText)).toEqual(durable.map(chatMessageText))
})

it('keeps a settled segment the refreshed fold does not carry yet', () => {
  const merged = preserveLocalPendingTurnMessages(
    [
      { id: '1-user', role: 'user', rowId: 1, parts: [{ type: 'text', text: 'q' }] },
      { id: '2-assistant', role: 'assistant', rowId: 2, parts: [{ type: 'text', text: 'First segment.' }] }
    ],
    [
      { id: 'user-1', role: 'user', rowId: 1, parts: [{ type: 'text', text: 'q' }] },
      {
        id: 'assistant-stream-1',
        role: 'assistant',
        pending: false,
        parts: [{ type: 'text', text: 'First segment.' }]
      },
      {
        id: 'assistant-stream-2',
        role: 'assistant',
        pending: false,
        parts: [{ type: 'text', text: 'Second segment.' }]
      }
    ]
  )

  expect(merged.map(chatMessageText)).toEqual(['q', 'First segment.', 'Second segment.'])
})
