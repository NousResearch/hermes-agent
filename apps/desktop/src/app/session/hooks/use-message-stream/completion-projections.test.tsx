import type { GatewayEventName } from '@hermes/shared'
import { act, cleanup } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { chatMessageText, textPart } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'

import { appendLiveSessionProjection } from '../use-session-actions/utils'

import { renderMessageStream } from './test-harness'

const SID = 'completion-projections'
const STORED = 'completion-projections-stored'

beforeEach(() => vi.useFakeTimers())
afterEach(() => {
  cleanup()
  vi.clearAllTimers()
  vi.useRealTimers()
  vi.restoreAllMocks()
})

function mount() {
  const initial = createClientSessionState(STORED, [
    { id: 'user-current', role: 'user', parts: [textPart('Inspect the file')] }
  ])

  const stream = renderMessageStream(SID, { states: new Map([[SID, initial]]) })

  const send = (type: GatewayEventName, payload: Record<string, unknown> = {}) =>
    act(() => stream.handleEvent({ type, payload, session_id: SID }))

  return { stream, send }
}

it.each([false, true])('settles before a queued next-turn prompt (fallback=%s)', async fallback => {
  const { stream, send } = mount()
  await send('message.start')
  await send('message.delta', { text: 'Current partial' })
  await act(async () => {
    await vi.advanceTimersByTimeAsync(100)
  })
  const current = stream.state()
  const streamId = current.streamId
  expect(streamId).not.toBeNull()

  // Use the same queue projection as session.resume; the pending response
  // still belongs to the current turn, not to this next-turn user row.
  const projected = appendLiveSessionProjection(current.messages, {
    session_id: SID,
    queued: { user: 'Run this next' }
  })

  stream.states.set(SID, { ...current, messages: projected, streamId: fallback ? null : streamId })
  const timeline = () => stream.state().messages.map(message => [message.role, chatMessageText(message)])

  const expected = [
    ['user', 'Inspect the file'],
    ['assistant', 'Current complete answer'],
    ['user', 'Run this next']
  ]

  await send('message.complete', { text: 'Current complete answer' })
  expect(timeline()).toEqual(expected)
  expect(stream.state().messages.find(message => message.id === streamId)?.pending).toBe(false)
  const ids = stream.state().messages.map(message => message.id)
  await send('message.complete', { text: 'Current complete answer' })
  expect(timeline()).toEqual(expected)
  expect(stream.state().messages.map(message => message.id)).toEqual(ids)

  // A real next-turn start with no delta must not reuse the previous answer,
  // even if the backend hasn't replaced the projected queued-user row yet.
  await send('message.start')
  await send('message.complete', { text: 'Current complete answer' })
  expect(timeline()).toEqual([...expected, ['assistant', 'Current complete answer']])
})

it.each([
  { partial: 'Partial answer', final: 'Partial answer' },
  { partial: 'Partial ans', final: 'Partial answer from the retained buffer' },
  { partial: 'Checking the file.', final: 'Checking the file.' }
])('merges the cumulative partial-error buffer once ($partial → $final)', async fixture => {
  const { stream, send } = mount()
  await send('message.start')
  // display.interim_assistant_messages=false keeps these responses in one
  // bubble; the gateway's terminal error carries the whole inflight buffer.
  await send('message.delta', { text: 'Checking the file.' })
  await send('tool.start', { name: 'read_file', tool_id: 'read-1', args: {} })
  await send('tool.complete', { name: 'read_file', tool_id: 'read-1', result: 'fixture' })
  await send('message.delta', { text: `\n\n${fixture.partial}` })

  const finalText = `Checking the file.\n\n${fixture.final}`

  const completion = {
    status: 'error',
    error: 'dispatcher failure',
    partial: true,
    recoverable: true,
    text: finalText
  }

  await send('message.complete', completion)
  const assistants = () => stream.state().messages.filter(message => message.role === 'assistant')
  expect(assistants().map(chatMessageText).join('')).toBe(finalText)
  expect(assistants()).toHaveLength(1)
  expect(assistants()[0]).toMatchObject({ error: 'dispatcher failure', pending: false })
  expect(assistants()[0].parts.filter(part => part.type === 'tool-call')).toMatchObject([
    { toolCallId: 'read-1', completedAt: expect.any(Number), result: 'fixture' }
  ])
  const ids = stream.state().messages.map(message => message.id)
  await send('message.complete', completion)
  expect(assistants().map(chatMessageText).join('')).toBe(finalText)
  expect(stream.state().messages.map(message => message.id)).toEqual(ids)
})
