import type { GatewayEvent } from '@hermes/shared'
// A `tool.complete` lost to a degraded websocket (fanout backlog overflow
// detaches the peer; reconnect, profile swap, hidden window) leaves its part
// without a `result`. `sealOpenToolParts` then stamps `completedAt` so the row
// stops spinning — which is right for the spinner but renders the row as
// "Result unavailable" even though the RESULT IS ON DISK: the runtime appends
// and flushes the tool result row BEFORE it projects `tool.completed`
// (agent/tool_executor.py), so a lost projection never means a lost result.
//
// The settle path must therefore hydrate stored history when it sealed a tool
// part, instead of leaving the transcript claiming the step produced nothing.
import { act, cleanup } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { type MessageStreamHarness, renderMessageStream } from './test-harness'

const SID = 'lost-tool-complete-session'

let stream: MessageStreamHarness

const emit = (event: GatewayEvent) => act(() => stream.handleEvent(event))

const toolParts = () =>
  stream
    .state(SID)
    .messages.flatMap(message => message.parts)
    .filter(part => part.type === 'tool-call')

describe('turn settles after a lost tool.complete', () => {
  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('hydrates stored history so the sealed row is not left as "Result unavailable"', async () => {
    const hydrateFromStoredSession = vi.fn(async () => undefined)
    stream = renderMessageStream(SID, { hydrateFromStoredSession })

    await act(async () => {
      await Promise.resolve()
    })

    emit({ payload: {}, session_id: SID, type: 'message.start' })
    emit({
      payload: { args: { command: 'pnpm test' }, name: 'terminal', tool_id: 'call-1' },
      session_id: SID,
      type: 'tool.start'
    })

    // No tool.complete arrives — the peer was detached mid-turn.
    emit({ payload: { text: 'Test run finished.' }, session_id: SID, type: 'message.complete' })

    const [sealed] = toolParts()
    expect(sealed?.type).toBe('tool-call')

    if (sealed?.type !== 'tool-call') {
      throw new Error('missing tool part')
    }

    // The row is sealed (no spinner) …
    expect(sealed.completedAt).toBeDefined()
    // … and the result is missing, which is exactly the state that renders as
    // "Result unavailable" — so the settle path MUST go back to stored history.
    expect(sealed.result).toBeUndefined()
    expect(hydrateFromStoredSession).toHaveBeenCalled()
  })

  it('does not force a hydrate when every tool part already carries its result', async () => {
    const hydrateFromStoredSession = vi.fn(async () => undefined)
    stream = renderMessageStream(SID, { hydrateFromStoredSession })

    await act(async () => {
      await Promise.resolve()
    })

    emit({ payload: {}, session_id: SID, type: 'message.start' })
    emit({
      payload: { args: { command: 'pnpm test' }, name: 'terminal', tool_id: 'call-1' },
      session_id: SID,
      type: 'tool.start'
    })
    emit({
      payload: { name: 'terminal', result: { output: 'ok' }, tool_id: 'call-1' },
      session_id: SID,
      type: 'tool.complete'
    })
    emit({ payload: { text: 'Done.' }, session_id: SID, type: 'message.complete' })

    const [settled] = toolParts()

    if (settled?.type !== 'tool-call') {
      throw new Error('missing tool part')
    }

    expect(settled.result).toEqual({ output: 'ok' })
    expect(hydrateFromStoredSession).not.toHaveBeenCalled()
  })
})
