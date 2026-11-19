import type { GatewayEvent } from '@hermes/shared'
// Repro for #119543: user sends a new message while the previous turn's
// `background_review` fork is still in flight. The fork itself emits nothing
// to the desktop, but the supersede window is where the previous turn's own
// frames can arrive reordered: a delta that lands after that turn settled
// still sits in the per-session queue when the new turn starts. Flushing it
// then seeds a bubble the new turn inherits (or that settles beside it),
// painting a stale duplicate of the previous reply into the transcript.
// Restart clears it because it was never persisted.
//
// Spec: bytes queued BEFORE `message.start` while no turn is live are
// orphans of a dead attempt and must be dropped at the boundary — never
// materialized into the transcript.
import { act, cleanup } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { chatMessageText } from '@/lib/chat-messages'

import { type MessageStreamHarness, renderMessageStream } from './test-harness'
import { STREAM_DELTA_FLUSH_MS } from './utils'

const SID = 'background-review-interrupt-session'
const STALE = 'previous answer fragment'
const LIVE = 'new reply'

let stream: MessageStreamHarness

async function mountHarness() {
  vi.useFakeTimers()
  stream = renderMessageStream(SID)
  await act(async () => {
    await Promise.resolve()
  })
}

const flushDeltas = async () => {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(STREAM_DELTA_FLUSH_MS)
  })
}

const emit = (event: GatewayEvent) => act(() => stream.handleEvent(event))

const assistantTexts = () =>
  (stream.state()?.messages ?? [])
    .filter(message => message.role === 'assistant' && !message.hidden)
    .map(message => chatMessageText(message))

describe('background-review interrupt must not paint stale bytes into the next turn (#119543)', () => {
  afterEach(() => {
    cleanup()
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it('drops a superseded-turn straggler queued before the new message.start', async () => {
    await mountHarness()

    // Turn A runs to completion and settles.
    emit({ payload: {}, session_id: SID, type: 'message.start' })
    emit({ payload: { text: STALE }, session_id: SID, type: 'message.delta' })
    await flushDeltas()
    emit({ payload: { text: STALE }, session_id: SID, type: 'message.complete' })
    expect(assistantTexts()).toEqual([STALE])
    expect(stream.state()?.streamId).toBeNull()

    // A reordered delta from turn A lands in the queue after turn A settled
    // and has not flushed yet when the user sends the next message.
    emit({ payload: { text: STALE }, session_id: SID, type: 'message.delta' })

    // New turn starts: the orphan must be dropped at the boundary, not
    // painted. Then the aborted review's turn-end signal races in.
    emit({ payload: {}, session_id: SID, type: 'message.start' })
    emit({ payload: { running: false }, session_id: SID, type: 'session.info' })

    // Turn B streams and completes normally.
    emit({ payload: { text: LIVE }, session_id: SID, type: 'message.delta' })
    await flushDeltas()
    emit({ payload: { text: LIVE }, session_id: SID, type: 'message.complete' })

    // Exactly turn A and turn B — no stale bubble seeded between them and no
    // stale fragment merged into the new reply.
    expect(assistantTexts()).toEqual([STALE, LIVE])
  })

  it('still flushes a live previous turn at message.start (steer keeps its text)', async () => {
    await mountHarness()

    // Turn A is still streaming when the next message.start arrives (steer /
    // chained start mid-turn): its queued bytes are live output of the bubble
    // on screen and must be kept.
    emit({ payload: {}, session_id: SID, type: 'message.start' })
    emit({ payload: { text: 'first half ' }, session_id: SID, type: 'message.delta' })
    emit({ payload: {}, session_id: SID, type: 'message.start' })

    expect(chatMessageText(stream.state()?.messages.at(-1)!)).toBe('first half ')
  })
})
