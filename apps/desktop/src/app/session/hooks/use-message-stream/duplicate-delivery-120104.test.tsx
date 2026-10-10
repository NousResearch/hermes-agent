import type { GatewayEventName } from '@hermes/shared'
import { act, cleanup } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { chatMessageText, textPart } from '@/lib/chat-messages'

import { renderMessageStream } from './test-harness'

const SID = 'issue-120104-duplicate-delivery'

afterEach(cleanup)

type Frame = [GatewayEventName, Record<string, unknown>]

const SKILL_TURN: Frame[] = [
  ['message.start', {}],
  ['message.delta', { text: 'reply text' }],
  ['tool.start', { name: 'skill_view', tool_id: 's1', args: { name: 'x' } }],
  ['tool.complete', { name: 'skill_view', tool_id: 's1', result: 'skill content' }],
  ['message.interim', { text: 'reply text', already_streamed: true }],
  ['message.complete', { text: 'reply text' }]
]

async function mount() {
  const stream = renderMessageStream(SID)

  const send = (type: GatewayEventName, payload: Record<string, unknown> = {}) =>
    act(() => stream.handleEvent({ type, payload, session_id: SID }))

  return { stream, send }
}

function expectSingleBubble(stream: ReturnType<typeof renderMessageStream>) {
  const messages = stream.state().messages.filter(m => m.role === 'assistant' && !m.hidden)
  expect(messages).toHaveLength(1)
  expect(chatMessageText(messages[0])).toBe('reply text')
  expect(messages[0].parts.filter(part => part.type === 'tool-call')).toHaveLength(1)
}

// A duplicate terminal frame must be idempotent even when the settled reply's
// text precedes its tool call. A late interim must not change that ownership.
it.each([false, true])('keeps a redelivered tool-turn completion idempotent (lateInterim=%s)', async lateInterim => {
  const { stream, send } = await mount()

  for (const [type, payload] of SKILL_TURN) {
    await send(type, payload)
  }

  expectSingleBubble(stream)
  const settled = stream.state().messages[0]

  if (lateInterim) {
    await send(...SKILL_TURN[4])
    expectSingleBubble(stream)
    expect(stream.state().messages[0]).toEqual(settled)
  }

  await send(...SKILL_TURN[5])
  expectSingleBubble(stream)
  expect(stream.state().messages[0]).toEqual(settled)

  const persistedTurn = {
    row_ids: [71, 72],
    user_row_id: 71,
    final_assistant_row_id: 72,
    complete: true
  }

  await send('message.complete', { text: 'reply text', persisted_turn: persistedTurn })
  expectSingleBubble(stream)
  const upgraded = stream.state().messages[0]
  expect(upgraded).toMatchObject({
    id: settled.id,
    completedAt: settled.completedAt,
    persistedTurn,
    durableComplete: true,
    rowId: persistedTurn.final_assistant_row_id
  })

  await send(...SKILL_TURN[5])
  expectSingleBubble(stream)
  expect(stream.state().messages[0]).toEqual(upgraded)
})

it.each(['message.start', 'running=true'])('keeps a same-text next turn after heartbeat settlement (%s)', async acceptance => {
  const hydrateFromStoredSession = vi.fn(() => new Promise<void>(() => {}))

  const stream = renderMessageStream(SID, { hydrateFromStoredSession })

  const send = (type: GatewayEventName, payload: Record<string, unknown> = {}) =>
    act(() => stream.handleEvent({ type, payload, session_id: SID }))

  for (const [type, payload] of SKILL_TURN) {
    await send(type, payload)
  }

  expectSingleBubble(stream)
  const settled = stream.state().messages[0]

  if (acceptance === 'message.start') {
    await send('message.start')
  } else {
    await send('session.info', { running: true })
  }

  await send('session.info', { running: false })
  expect(stream.state()).toMatchObject({ busy: false, turnLive: false, heartbeatSettledStreamId: null })
  expect(hydrateFromStoredSession).toHaveBeenCalled()

  await send('message.complete', { text: 'reply text' })
  const replies = stream.state().messages.filter(message => message.role === 'assistant' && !message.hidden)
  expect(replies.map(chatMessageText)).toEqual(['reply text', 'reply text'])
  expect(replies[0]).toEqual(settled)
  expect(replies[1].id).not.toBe(settled.id)

  await send('message.complete', { text: 'reply text' })
  expect(stream.state().messages.filter(message => message.role === 'assistant' && !message.hidden)).toEqual(replies)
})

it('keeps same-text replies distinct across turn, user and durable-row boundaries', async () => {
  const receipt = (rowId: number) => ({
    row_ids: [rowId - 1, rowId],
    user_row_id: rowId - 1,
    final_assistant_row_id: rowId,
    complete: true
  })

  for (const boundary of ['turn', 'user', 'durable-row']) {
    const { stream, send } = await mount()

    for (const [type, payload] of SKILL_TURN.slice(0, -1)) {
      await send(type, payload)
    }

    await send('message.complete', { text: 'reply text', persisted_turn: receipt(72) })
    expectSingleBubble(stream)
    const settled = stream.state().messages[0]

    if (boundary === 'turn') {
      await send('message.start')
    } else if (boundary === 'user') {
      stream.states.set(SID, {
        ...stream.state(),
        messages: [...stream.state().messages, { id: 'next-user', role: 'user', parts: [textPart('Again.')] }]
      })
    }

    await send('message.complete', {
      text: 'reply text',
      ...(boundary === 'durable-row' ? { persisted_turn: receipt(74) } : {})
    })

    const replies = stream.state().messages.filter(message => message.role === 'assistant' && !message.hidden)
    expect(replies.map(chatMessageText), boundary).toEqual(['reply text', 'reply text'])
    expect(replies[0], boundary).toEqual(settled)
    expect(replies[1].id, boundary).not.toBe(settled.id)
    cleanup()
  }
})
