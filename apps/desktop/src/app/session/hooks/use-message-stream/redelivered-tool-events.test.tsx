import type { GatewayEventName } from '@hermes/shared'
// A tool event can reach the renderer twice. A background `delegate_task`
// returns in about a second, and a second copy of its start or completion
// used to draw the whole fan-out again: "5 subagents" showed 10 rows. A copy
// of a call this turn already finished must change nothing. llama.cpp's
// constant ids and a later turn reusing an id still get their own rows.
import { act, cleanup } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { type MessageStreamHarness, renderMessageStream } from './test-harness'

const SID = 'redelivered-tool-events-session'
const CALL = 'toolu_delegate'
const TEXT = 'Dispatching 5 reviewers now.'
const goals = [1, 2, 3, 4, 5].map(index => `review packet ${index}`)
const args = { tasks: goals.map(goal => ({ goal })) }
const result = JSON.stringify({ count: 5, goals, mode: 'background', status: 'dispatched' })

let stream: MessageStreamHarness

const event = (type: GatewayEventName, timestamp: number, payload: Record<string, unknown> = {}) =>
  act(() => stream.handleEvent({ payload: { ...payload, timestamp }, session_id: SID, type }))

const start = (timestamp: number) => event('tool.start', timestamp, { args, name: 'delegate_task', tool_id: CALL })

const complete = (timestamp: number) =>
  event('tool.complete', timestamp, { name: 'delegate_task', result, tool_id: CALL })

const rows = (toolCallId = CALL) =>
  (stream.state(SID).messages ?? []).flatMap(message =>
    message.parts.flatMap(part => (part.type === 'tool-call' && part.toolCallId === toolCallId ? [part] : []))
  )

function dispatch(seal: 'after-tools' | 'before-tools' | 'none') {
  event('message.start', 100)
  event('message.delta', 101, { text: TEXT })

  if (seal === 'before-tools') {
    event('message.interim', 102, { text: TEXT })
  }

  start(103)
  complete(104.5)

  if (seal === 'after-tools') {
    event('message.interim', 105, { text: TEXT })
  }
}

describe('a re-delivered event for a call this turn already finished', () => {
  beforeEach(() => {
    stream = renderMessageStream(SID)
  })

  afterEach(() => {
    cleanup()
  })

  for (const seal of ['none', 'after-tools', 'before-tools'] as const) {
    it(`draws the delegation once when its start and completion arrive again (interim seal: ${seal})`, () => {
      dispatch(seal)
      start(106)
      complete(107)
      event('message.delta', 108, { text: "While they work, I'll map the architecture myself." })

      const found = rows()
      expect(found).toHaveLength(1)
      // The copy neither reopened the row nor restamped how long it ran.
      expect(found[0]).toMatchObject({ completedAt: 104.5, result, timestamp: 103 })
    })
  }

  it('keeps one row when a sealed call completes twice after an interim (#140397)', () => {
    event('message.start', 100)
    event('tool.start', 101, { args: { url: 'https://example.test' }, name: 'browser', tool_id: 'call_page' })
    event('message.interim', 102, { text: 'Reading the page.' })
    event('tool.complete', 103, { name: 'browser', result: 'page output', tool_id: 'call_page' })
    event('tool.complete', 104, { name: 'browser', result: 'page output', tool_id: 'call_page' })

    const found = rows('call_page')
    expect(found).toHaveLength(1)
    expect(found[0]).toMatchObject({ completedAt: 103, result: 'page output' })
    expect(stream.state(SID).streamId).toBeNull()
  })

  it('still draws a constant-id call with different arguments in the same turn as its own row', () => {
    event('message.start', 200)
    event('tool.start', 201, { args: { command: 'echo one' }, name: 'terminal', tool_id: 'call_const' })
    event('tool.complete', 202, { name: 'terminal', result: 'one', tool_id: 'call_const' })
    event('tool.start', 203, { args: { command: 'echo two' }, name: 'terminal', tool_id: 'call_const' })
    event('tool.complete', 204, { name: 'terminal', result: 'two', tool_id: 'call_const' })

    expect(rows('call_const').map(part => part.result)).toEqual(['one', 'two'])
  })

  it('still draws a later turn that repeats the same call as its own row', () => {
    dispatch('none')
    event('message.complete', 110, { text: TEXT })
    act(() => {
      const state = stream.state(SID)
      stream.states.set(SID, {
        ...state,
        messages: [
          ...state.messages,
          { id: 'user-2', parts: [{ text: 'again', type: 'text' }], role: 'user', timestamp: 111 }
        ]
      })
    })

    event('message.start', 112)
    start(113)
    complete(114)

    expect(rows()).toHaveLength(2)
  })
})
