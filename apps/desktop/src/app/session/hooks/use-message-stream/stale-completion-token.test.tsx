// Stale-completion drop (#119543): a late message.complete carrying a turn
// token this window already superseded (or that mismatches the live claim)
// must not clear prompts, flush the queue, or append after the next user row.
import type { GatewayEventName } from '@hermes/shared'
import { act, cleanup } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { textPart } from '@/lib/chat-messages'

import { appendMidTurnUserMessage } from '../use-prompt-actions/rewind'

import type { MessageStreamHarness } from './test-harness'
import { renderMessageStream } from './test-harness'
import { STREAM_DELTA_FLUSH_MS } from './utils'

const SID = 'stale-completion-token'

let stream: MessageStreamHarness

async function mountHarness() {
  vi.useFakeTimers()
  stream = renderMessageStream(SID)
  await act(async () => {
    await Promise.resolve()
  })
}

const emit = (event: GatewayEventName, payload: Record<string, unknown> = {}) =>
  act(() => stream.handleEvent({ type: event, payload, session_id: SID }))

const armSeed = (superseded: string) => {
  const state = stream.state(SID)
  stream.states.set(SID, {
    ...state,
    supersededTurnToken: superseded,
    turnToken: null
  })
}

describe('stamped message.complete turn identity', () => {
  afterEach(() => {
    cleanup()
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it('drops a complete whose turn matches the superseded seed token', async () => {
    await mountHarness()

    await emit('message.start', { turn: 'turn-old' })
    await emit('message.delta', { text: 'old partial' })
    await act(async () => {
      await vi.advanceTimersByTimeAsync(STREAM_DELTA_FLUSH_MS)
    })

    // User seeds a new turn: arms superseded to the previous token.
    armSeed('turn-old')
    stream.states.set(
      SID,
      appendMidTurnUserMessage(stream.state(SID), {
        id: 'user-next',
        role: 'user',
        parts: [textPart('next question')]
      })
    )

    // Late frame from the superseded turn arrives after the seed.
    await emit('message.complete', { text: 'stale fragment', turn: 'turn-old' })
    await act(async () => {
      await vi.advanceTimersByTimeAsync(STREAM_DELTA_FLUSH_MS)
    })

    const messages = stream.state(SID).messages
    expect(messages.at(-1)?.role).toBe('user')
    expect(messages.flatMap(m => m.parts).some(p => p.type === 'text' && p.text.includes('stale fragment'))).toBe(false)
    // Side effects must not run either: stream bookkeeping stays seeded.
    expect(stream.state(SID).turnToken).toBeNull()
    expect(stream.state(SID).supersededTurnToken).toBe('turn-old')
  })

  it('drops a complete that mismatches an already-claimed live turn token', async () => {
    await mountHarness()

    await emit('message.start', { turn: 'turn-live' })
    await emit('message.delta', { text: 'live partial' })
    await act(async () => {
      await vi.advanceTimersByTimeAsync(STREAM_DELTA_FLUSH_MS)
    })

    // A different turn's late complete must not settle this stream.
    await emit('message.complete', { text: 'other turn answer', turn: 'turn-other' })

    expect(stream.state(SID).streamId).toBeTruthy()
    expect(stream.state(SID).awaitingResponse).toBe(false)
    expect(stream.text(SID)).toContain('live partial')
    expect(
      stream.state(SID).messages.flatMap(m => m.parts).some(p => p.type === 'text' && p.text.includes('other turn answer'))
    ).toBe(false)
  })

  it('keeps a matching stamped complete for the in-flight turn', async () => {
    await mountHarness()

    await emit('message.start', { turn: 'turn-live' })
    await emit('message.delta', { text: 'partial answer' })
    await act(async () => {
      await vi.advanceTimersByTimeAsync(STREAM_DELTA_FLUSH_MS)
    })
    await emit('message.complete', { text: 'final answer', turn: 'turn-live' })

    const tail = stream.state(SID).messages.at(-1)
    expect(tail?.role).toBe('assistant')
    expect(tail?.pending).toBe(false)
    expect(tail?.parts).toMatchObject([{ type: 'text', text: 'final answer' }])
    expect(stream.state(SID).streamId).toBeNull()
  })

  it('keeps an unstamped complete (older gateways / muted emitters)', async () => {
    await mountHarness()

    // Seed arm without a stamped start — pre-token compatibility path.
    armSeed('turn-old')
    await emit('message.start')
    await emit('message.delta', { text: 'live without token' })
    await act(async () => {
      await vi.advanceTimersByTimeAsync(STREAM_DELTA_FLUSH_MS)
    })
    await emit('message.complete', { text: 'done without token' })

    const tail = stream.state(SID).messages.at(-1)
    expect(tail?.role).toBe('assistant')
    expect(tail?.parts).toMatchObject([{ type: 'text', text: 'done without token' }])
    expect(stream.state(SID).streamId).toBeNull()
  })

  it('discards residual queued deltas on message.start (not flush)', async () => {
    await mountHarness()

    await emit('message.start', { turn: 'turn-old' })
    // Residual deltas still in the flush queue for this session.
    act(() => stream.appendDelta(SID, 'RESIDUAL-STALE-'))

    // New turn begins — queue must be discarded, not applied to a new bubble.
    await emit('message.start', { turn: 'turn-new' })
    await act(async () => {
      await vi.advanceTimersByTimeAsync(STREAM_DELTA_FLUSH_MS)
    })

    expect(stream.text(SID)).not.toContain('RESIDUAL-STALE-')
    expect(stream.state(SID).turnToken).toBe('turn-new')
  })
})
