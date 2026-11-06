import { act, cleanup } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'

import { chatMessageText, toChatMessages } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'
import type { SessionMessage } from '@/types/hermes'

import { renderMessageStream } from './test-harness'

afterEach(cleanup)

const SID = 'hydrated-completion-identity'

const history: SessionMessage[] = [
  { id: 1, role: 'user', content: 'Inspect the fixture.' },
  {
    id: 2,
    role: 'assistant',
    content: 'Checking the fixture.',
    tool_calls: [{ id: 'read-1', type: 'function', function: { name: 'read_file', arguments: '{}' } }]
  },
  { id: 3, role: 'tool', content: 'fixture read', tool_call_id: 'read-1' },
  { id: 4, role: 'assistant', content: 'Verified.' }
]

it.each([false, true])('settles a hydrated durable reply by identity with boundary=%s', async boundary => {
  const messages = toChatMessages(history)

  const stream = renderMessageStream(SID, {
    states: new Map([
      [
        SID,
        { ...createClientSessionState('stored-identity', messages), interimBoundaryPending: boundary, turnLive: true }
      ]
    ])
  })

  const before = messages.map(message => message.id)

  await act(() =>
    stream.handleEvent({
      type: 'message.complete',
      session_id: SID,
      payload: {
        text: 'Verified.',
        persisted_turn: { row_ids: [1, 2, 3, 4], user_row_id: 1, final_assistant_row_id: 4, complete: true }
      }
    })
  )

  expect(stream.state().messages.map(message => message.id)).toEqual(before)
  expect(stream.state().messages.map(chatMessageText)).toEqual(messages.map(chatMessageText))
})

it('keeps an identical later reply with a different durable row identity', async () => {
  const messages = toChatMessages(history)

  const stream = renderMessageStream(SID, {
    states: new Map([
      [SID, { ...createClientSessionState('stored-identity', messages), interimBoundaryPending: true, turnLive: true }]
    ])
  })

  await act(() =>
    stream.handleEvent({
      type: 'message.complete',
      session_id: SID,
      payload: {
        text: 'Verified.',
        persisted_turn: { row_ids: [1, 2, 3, 4, 5], user_row_id: 1, final_assistant_row_id: 5, complete: true }
      }
    })
  )

  expect(stream.state().messages).toHaveLength(messages.length + 1)
  expect(stream.state().messages.at(-1)?.rowId).toBe(5)
})
