import type { GatewayEventName } from '@hermes/shared'
import { act, cleanup } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'

import { graftRefreshedTailOntoBackfill } from '@/app/chat/transcript-backfill'
import type { ClientSessionState } from '@/app/types'
import { chatMessageText, toChatMessages } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'
import type { SessionMessage } from '@/types/hermes'

import { preserveLocalPendingTurnMessages } from '../use-session-actions/utils'

import { renderMessageStream } from './test-harness'

afterEach(cleanup)

const FINAL = 'Here is the final answer to your question.'

const PRIOR: SessionMessage[] = [
  { id: 1, role: 'user', content: 'earlier', timestamp: 1 },
  { id: 2, role: 'assistant', content: 'earlier answer', timestamp: 2 },
  { id: 3, role: 'user', content: 'q', timestamp: 3 }
]

const TOOL_ROUND: SessionMessage[] = [
  {
    id: 4,
    role: 'assistant',
    content: '',
    timestamp: 4,
    tool_calls: [{ id: 't1', type: 'function', function: { name: 'read_file', arguments: '{}' } }]
  },
  { id: 5, role: 'tool', content: 'ok', timestamp: 5, tool_call_id: 't1', tool_name: 'read_file' }
]

function mount(sid: string, prior: SessionMessage[] = PRIOR) {
  const states = new Map<string, ClientSessionState>()
  states.set(sid, { ...createClientSessionState(), messages: toChatMessages(prior) })
  const stream = renderMessageStream(sid, { states })

  const send = (type: GatewayEventName, payload: Record<string, unknown> = {}) =>
    act(() => stream.handleEvent({ type, payload, session_id: sid }))

  // The composition every history refresh applies (use-background-sync).
  const refresh = (rows: SessionMessage[]) =>
    act(() => {
      const local = stream.state().messages
      const next = preserveLocalPendingTurnMessages(graftRefreshedTailOntoBackfill(toChatMessages(rows), local), local)
      states.set(sid, { ...stream.state(), messages: next })
    })

  const flush = () => act(() => new Promise(resolve => setTimeout(resolve, 50)))

  const copies = () => stream.state().messages.filter(message => chatMessageText(message).includes(FINAL))

  return { copies, flush, refresh, send, stream }
}

// #123801: one stored row, one completion, two roots — `timestamp-index-assistant`
// and `assistant-stream-*`. A reconcile landing between the gateway commit and
// the end of the live stream folds the live bubble into the committed row; the
// turn's tail then re-seeds a bubble under the surviving stream id.
it.each([
  ['a plain reply', [] as SessionMessage[]],
  ['a reply after a tool round', TOOL_ROUND]
])('%s paints once when history lands before the stream tail', async (_label, toolRows) => {
  const { copies, flush, refresh, send } = mount(`twin-${toolRows.length}`)

  await send('message.start')

  if (toolRows.length) {
    await send('tool.start', { name: 'read_file', tool_id: 't1', args: {} })
    await send('tool.complete', { name: 'read_file', tool_id: 't1', result: 'ok' })
  }

  await send('message.delta', { text: `\n\n${FINAL.slice(0, 20)}` })
  await flush()
  await refresh([...PRIOR, ...toolRows, { id: 6, role: 'assistant', content: FINAL, timestamp: 6 }])
  await send('message.delta', { text: FINAL.slice(20) })
  await flush()
  await send('message.complete', {
    text: FINAL,
    persisted_turn: {
      row_ids: [3, ...toolRows.map(row => row.id), 6],
      user_row_id: 3,
      final_assistant_row_id: 6,
      complete: true
    }
  })

  expect(copies()).toHaveLength(1)
  expect(copies()[0]).toMatchObject({ pending: false, rowId: 6, durableComplete: true })
  expect(copies()[0].id).not.toMatch(/^assistant-stream-/)
})

it('never settles onto an identical reply from an earlier occurrence', async () => {
  const prior: SessionMessage[] = [
    { id: 1, role: 'user', content: 'earlier', timestamp: 1 },
    { id: 2, role: 'assistant', content: FINAL, timestamp: 2 },
    { id: 3, role: 'user', content: 'again', timestamp: 3 }
  ]

  const { copies, flush, send, stream } = mount('twin-earlier-occurrence', prior)

  await send('message.start')
  await send('message.delta', { text: FINAL })
  await flush()
  await send('message.complete', {
    text: FINAL,
    persisted_turn: { row_ids: [3, 4], user_row_id: 3, final_assistant_row_id: 4, complete: true }
  })

  expect(copies().map(message => message.rowId)).toEqual([2, 4])
  expect(stream.state().messages.at(-1)?.id).toMatch(/^assistant-stream-/)
})
