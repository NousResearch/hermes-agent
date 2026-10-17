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
 * with no live turn is the discriminator; the settled turn itself must keep
 * its bubble and a new turn must still paint normally.
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
