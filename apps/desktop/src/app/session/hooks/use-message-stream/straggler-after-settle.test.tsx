import type { GatewayEvent } from '@hermes/shared'
import { act, cleanup } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { chatMessageText } from '@/lib/chat-messages'

import { type MessageStreamHarness, renderMessageStream } from './test-harness'

/**
 * Stream frames that arrive after their turn settled must not seed a bubble.
 *
 * The event stream can reorder or replay frames across a settle
 * (`message.complete`, or a `running=false` heartbeat): a late `message.delta`
 * or a tool event whose call never streamed here lands with no live turn.
 * Seeding then paints a local-only `assistant-stream-…` bubble beside the
 * settled reply, and every merge skips `pending` rows (background refresh,
 * resume reconcile, overlay), so hydration can never retire it (#127665).
 * A genuinely new turn always opens with `message.start`, so an idle session
 * with no live turn is the discriminator.
 *
 * Boundaries pinned here:
 * - a heartbeat settle drains the turn's queued deltas into its own bubble
 *   (settled, text kept) and a later unowned tool frame seeds nothing;
 * - a session's first output still opens its bubble with an empty transcript;
 * - a legitimate late patch (a completion crossing the settle) still lands on
 *   its bubble without re-arming the stream;
 * - a new turn after a settled one still paints normally.
 */

const SID = 'session-1'

let stream: MessageStreamHarness

const ev = (type: string, payload: Record<string, unknown> = {}): GatewayEvent =>
  ({ payload, session_id: SID, type }) as GatewayEvent

const send = (type: string, payload: Record<string, unknown> = {}) => act(() => stream.handleEvent(ev(type, payload)))

const COMMITTED = { row_ids: [1, 2, 3], user_row_id: 1, final_assistant_row_id: 3, complete: true }

function runTurn() {
  send('message.start')
  send('message.delta', { text: 'the reply text' })
  send('tool.start', { name: 'terminal', tool_id: 'call-1', args: {} })
  send('tool.complete', { name: 'terminal', tool_id: 'call-1', result: 'ok' })
  send('message.complete', { text: 'the reply text', persisted_turn: COMMITTED })
}

const flushTimers = () => act(async () => void (await vi.advanceTimersByTimeAsync(100)))

const visibleAssistants = () => stream.state(SID).messages.filter(m => m.role === 'assistant' && !m.hidden)

beforeEach(() => {
  vi.useFakeTimers()
})

afterEach(() => {
  cleanup()
  vi.useRealTimers()
  vi.restoreAllMocks()
})

describe('useMessageStream straggler frames after settle (#127665)', () => {
  it('a late delta and an unowned tool frame seed no second bubble', async () => {
    stream = renderMessageStream(SID)

    runTurn()

    send('message.delta', { text: 'the reply text' })
    send('tool.start', { name: 'terminal', tool_id: 'call-9', args: {} })
    await flushTimers()

    const assistants = visibleAssistants()
    expect(assistants).toHaveLength(1)
    expect(chatMessageText(assistants[0])).toContain('the reply text')
  })

  it('a heartbeat settle keeps queued text in the turn bubble, no ghost seed', async () => {
    stream = renderMessageStream(SID)

    send('message.start')
    // Queued, not yet flushed when the turn ends.
    send('message.delta', { text: 'partial answer' })
    send('session.info', { running: false })

    await flushTimers()

    const assistants = visibleAssistants()
    expect(assistants).toHaveLength(1)
    expect(assistants[0].pending).toBe(false)
    expect(chatMessageText(assistants[0])).toContain('partial answer')

    // An unowned tool frame after the heartbeat settle seeds nothing either.
    send('tool.start', { name: 'terminal', tool_id: 'call-9', args: {} })
    await flushTimers()

    expect(visibleAssistants()).toHaveLength(1)
    expect(stream.state(SID).streamId).toBeNull()
  })

  it('a session with an empty transcript still opens its first bubble', async () => {
    stream = renderMessageStream(SID)

    send('message.delta', { text: 'first words' })
    await flushTimers()

    const assistants = visibleAssistants()
    expect(assistants).toHaveLength(1)
    expect(chatMessageText(assistants[0])).toContain('first words')
  })

  it('a legitimate completion crossing the settle still lands on its bubble', async () => {
    stream = renderMessageStream(SID)

    send('message.start')
    send('message.delta', { text: 'an answer' })
    send('tool.start', { name: 'terminal', tool_id: 'call-1', args: {} })
    // The turn's completion never saw the tool's result: settle-time sealing
    // closes the part without one.
    send('message.complete', { text: 'an answer', persisted_turn: COMMITTED })

    // The lost completion arrives after the settle and patches its own bubble.
    send('tool.complete', { name: 'terminal', tool_id: 'call-1', result: 'ok' })
    await flushTimers()

    const assistants = visibleAssistants()
    expect(assistants).toHaveLength(1)

    const toolPart = assistants[0].parts.find(part => part.type === 'tool-call' && part.toolCallId === 'call-1')
    expect(toolPart && 'result' in toolPart ? toolPart.result : undefined).toBe('ok')
    expect(stream.state(SID).streamId).toBeNull()
  })

  it('a new turn still paints after the previous one settled', async () => {
    stream = renderMessageStream(SID)

    runTurn()

    send('message.start')
    send('message.delta', { text: 'second answer' })
    send('message.complete', { text: 'second answer' })
    await flushTimers()

    const texts = visibleAssistants().map(chatMessageText)
    expect(texts).toHaveLength(2)
    expect(texts.at(-1)).toContain('second answer')
  })
})
