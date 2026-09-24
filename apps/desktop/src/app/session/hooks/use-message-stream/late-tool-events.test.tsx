import type { GatewayEventName } from '@hermes/shared'
// Tool events route to the bubble that owns the call id, not to whatever is
// streaming now. A result that lands AFTER its part was sealed (interim
// commentary, mid-turn user insert, turn settle) must re-attach to that part
// instead of seeding a duplicate row under "Result unavailable" (#113035) —
// but a call id that repeats across turns (llama.cpp emits one constant id)
// must never be routed back onto a finished row from an earlier turn.
import { act, cleanup } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { appendMidTurnUserMessage } from '@/app/session/hooks/use-prompt-actions/rewind'

import { type MessageStreamHarness, renderMessageStream } from './test-harness'

const SID = 'late-tool-events-session'

let stream: MessageStreamHarness

const event = (type: GatewayEventName, timestamp: number, payload: Record<string, unknown> = {}) =>
  act(() => stream.handleEvent({ payload: { ...payload, timestamp }, session_id: SID, type }))

const toolRows = (toolCallId: string) =>
  (stream.state(SID).messages ?? []).flatMap((message, messageIndex) =>
    message.parts.flatMap(part =>
      part.type === 'tool-call' && part.toolCallId === toolCallId ? [{ messageIndex, part }] : []
    )
  )

describe('tool events for a part that already exists on a sealed message', () => {
  beforeEach(() => {
    stream = renderMessageStream(SID)
  })

  afterEach(() => {
    cleanup()
  })

  it('attaches a late completion to the sealed bubble that owns the call', () => {
    event('message.start', 100)
    event('message.delta', 101, { text: 'Scraping the page.' })
    event('tool.start', 102, { args: { url: 'https://example.test' }, name: 'browser', tool_id: 'call-late-1' })
    // Interim commentary seals the streaming bubble while the browser scrape
    // keeps running (minutes in practice). streamId drops; the tool part is
    // sealed with completedAt and no result.
    event('message.interim', 103, { text: 'Scraping the page.' })
    event('tool.complete', 260.5, { name: 'browser', result: 'ok', tool_id: 'call-late-1' })

    const state = stream.state(SID)
    const assistants = (state.messages ?? []).filter(message => message.role === 'assistant')

    // The sealed part got the result instead of staying "Result unavailable"
    // and no duplicate row was seeded into a fresh bubble.
    expect(assistants).toHaveLength(1)
    expect(assistants[0].parts).toEqual([
      expect.objectContaining({ completedAt: 102, type: 'text' }),
      expect.objectContaining({ result: 'ok', toolCallId: 'call-late-1', type: 'tool-call' })
    ])
    expect(toolRows('call-late-1')).toHaveLength(1)
    // The late completion does not re-open a stream bubble.
    expect(state.streamId).toBeNull()
  })

  it('keeps one live row when the user types mid-turn and the running tool then completes', () => {
    event('message.start', 300)
    event('tool.start', 301, { args: { command: 'timeout 120 debug share' }, name: 'terminal', tool_id: 'call_02' })

    // Mid-turn user insert (steer): the desktop seals the live bubble and
    // clears streamId so the next assistant output lands below the user row.
    act(() => {
      stream.states.set(
        SID,
        appendMidTurnUserMessage(stream.state(SID), {
          id: 'user-steer',
          role: 'user',
          parts: [{ type: 'text', text: 'also check the log' }],
          timestamp: 302
        })
      )
    })

    // A running-phase event for the same call (progress/replay) must update
    // the existing row, not seed a second live row under the user message.
    event('tool.start', 303, { args: { command: 'timeout 120 debug share' }, name: 'terminal', tool_id: 'call_02' })

    const running = toolRows('call_02')
    expect(running).toHaveLength(1)
    expect(running[0].messageIndex).toBe(0)
    expect((running[0].part as { completedAt?: number }).completedAt).toBeUndefined()

    event('tool.complete', 304, { name: 'terminal', result: 'shared', tool_id: 'call_02' })

    const found = toolRows('call_02')
    expect(found).toHaveLength(1)
    expect(found[0].messageIndex).toBe(0)
    expect(found[0].part).toMatchObject({ result: 'shared', type: 'tool-call' })
    // user row stays the tail: no phantom assistant bubble was appended.
    expect(stream.state(SID).messages.at(-1)?.id).toBe('user-steer')
  })

  it('draws a reused tool_call_id from a later turn as its own row, not over the finished one', () => {
    event('message.start', 400)
    event('tool.start', 401, { args: { command: 'echo step1' }, name: 'terminal', tool_id: 'call_const' })
    event('tool.complete', 402, { name: 'terminal', result: 'step 1 output', tool_id: 'call_const' })
    event('message.complete', 403, { text: 'first' })

    // Same id in the next turn: a finished part is never the owner of a new
    // running event, so this call lands in the live bubble.
    event('message.start', 500)
    event('tool.start', 501, { args: { command: 'echo step2' }, name: 'terminal', tool_id: 'call_const' })
    event('tool.complete', 502, { name: 'terminal', result: 'step 2 output', tool_id: 'call_const' })
    event('message.complete', 503, { text: 'second' })

    const rows = toolRows('call_const')
    expect(rows).toHaveLength(2)
    expect(rows[0].part).toMatchObject({ args: { command: 'echo step1' }, result: 'step 1 output' })
    expect(rows[1].part).toMatchObject({ args: { command: 'echo step2' }, result: 'step 2 output' })
    expect(rows[1].messageIndex).toBeGreaterThan(rows[0].messageIndex)
  })
})
