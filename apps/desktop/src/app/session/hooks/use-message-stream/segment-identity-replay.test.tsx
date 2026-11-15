import { act, cleanup } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { chatMessageText } from '@/lib/chat-messages'

import { type MessageStreamHarness, renderMessageStream } from './test-harness'
import { STREAM_DELTA_FLUSH_MS } from './utils'

// A turn's frames carry their wire identity (tui_gateway/stream_segments.py):
// message.start the turn id, each delta its segment + offset, each interim the
// segment it seals. A frame re-sent under a new seq (reconnect replay, a
// superseded attempt re-streaming its segment) must upsert what is on screen.
const SID = 'segment-identity-replay'
const TURN = 'turn-a'
const COMMENTARY = 'Commentary before the tool, streamed once.'
const FINAL = 'A different final answer.'

let stream: MessageStreamHarness

const send = (type: string, payload: Record<string, unknown>) =>
  act(() => stream.handleEvent({ payload, session_id: SID, type: type as never }))

const start = (turn = TURN) => send('message.start', { turn_id: turn })

const delta = (text: string, segment: number, offset: number, turn = TURN) =>
  send('message.delta', { offset, segment, text, turn_id: turn })

const interim = (text: string, segment: number, turn = TURN) =>
  send('message.interim', { already_streamed: true, segment, text, turn_id: turn })

const tool = (phase: 'start' | 'complete', id = 'call-1') =>
  send(`tool.${phase}`, { name: 'terminal', tool_id: id, ...(phase === 'complete' ? { result: 'ok' } : {}) })

const flush = () => act(async () => void (await vi.advanceTimersByTimeAsync(STREAM_DELTA_FLUSH_MS + 10)))

const commentaryCopies = () =>
  stream
    .state()
    .messages.filter(message => message.role === 'assistant' && !message.hidden)
    .reduce((count, message) => count + chatMessageText(message).split(COMMENTARY).length - 1, 0)

describe('a re-sent segment of a stamped turn renders once', () => {
  afterEach(() => {
    cleanup()
    vi.useRealTimers()
  })

  it('drops the start + narration a superseded attempt re-sends when the tool starts', async () => {
    vi.useFakeTimers()
    stream = renderMessageStream(SID)
    await start()
    await delta(COMMENTARY.slice(0, 10), 0, 0)
    await delta(COMMENTARY.slice(10), 0, 10)
    await flush()
    await interim(COMMENTARY, 0)
    await tool('start')
    // The replay: the same start and narration deltas again.
    await start()
    await delta(COMMENTARY.slice(0, 10), 0, 0)
    await delta(COMMENTARY.slice(10), 0, 10)
    await interim(COMMENTARY, 0)
    await flush()
    await tool('complete')
    await delta(FINAL, 1, 0)
    await flush()
    await send('message.complete', { text: FINAL })

    expect(commentaryCopies()).toBe(1)
  })

  it('keeps commentary the agent genuinely repeats in a later segment', async () => {
    vi.useFakeTimers()
    stream = renderMessageStream(SID)
    await start()
    await delta(COMMENTARY, 0, 0)
    await flush()
    await interim(COMMENTARY, 0)
    await tool('start', 'call-1')
    await tool('complete', 'call-1')
    await delta(COMMENTARY, 1, 0)
    await flush()
    await interim(COMMENTARY, 1)
    await tool('start', 'call-2')
    await tool('complete', 'call-2')
    await delta(FINAL, 2, 0)
    await flush()
    await send('message.complete', { text: FINAL })

    expect(commentaryCopies()).toBe(2)
  })
})
