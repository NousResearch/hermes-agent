import type { GatewayEventName } from '@hermes/shared'
import { act, cleanup } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { renderMessageStream, type MessageStreamHarness } from './test-harness'

const SID = 'interrupted-latch-session'

let stream: MessageStreamHarness

const event = (type: GatewayEventName, timestamp: number, payload: Record<string, unknown> = {}) =>
  act(() => stream.handleEvent({ payload: { ...payload, timestamp }, session_id: SID, type }))

describe('interrupted-latch guard slots the turn after a stop', () => {
  beforeEach(() => {
    stream = renderMessageStream(SID)
  })

  afterEach(() => {
    cleanup()
  })

  it('arms and appends a turn whose message.start arrives after an interrupt was cleared', () => {
    event('message.start', 100)
    event('message.delta', 101, { text: 'long reply' })

    // Stop pressed: cancelRun writes interrupted=true, busy=false, streamId
    // null (the same local state writes the real stop button makes).
    act(() => {
      const s = stream.states.get(SID)!
      stream.states.set(SID, { ...s, busy: false, awaitingResponse: false, streamId: null, interrupted: true })
    })

    // Fresh submit clears the flag (submit.ts seedOptimistic writes
    // interrupted:false) and appends the optimistic user row.
    act(() => {
      const s = stream.states.get(SID)!
      stream.states.set(SID, {
        ...s,
        interrupted: false,
        busy: true,
        awaitingResponse: true,
        messages: [...s.messages, { id: 'user-warm', parts: [{ text: 'hi', type: 'text' }], role: 'user' } as never]
      })
    })

    // Backend accepts the new turn.
    event('message.start', 200)
    event('message.delta', 201, { text: 'fresh answer' })
    event('message.complete', 202, { text: 'fresh answer' })

    const state = stream.state(SID)
    expect(state.busy).toBe(false)
    expect(state.messages.some(m => m.role === 'user' && m.id === 'user-warm')).toBe(true)
    expect((state.messages ?? []).some(m => m.role === 'assistant' && JSON.stringify(m.parts).includes('fresh answer'))).toBe(true)
  })

  it('arms a plain chained turn whose message.start races the interrupt latch', () => {
    // Seeded interrupted=true WITHOUT a fresh submit — the backend-launched
    // chained-turn shape. This is the path the boolean latch cannot handle.
    event('message.start', 100)
    event('message.delta', 101, { text: 'goal reply' })
    event('message.complete', 102, { text: 'goal reply' })

    act(() => {
      const s = stream.states.get(SID)!

      stream.states.set(SID, { ...s, busy: false, awaitingResponse: false, streamId: null, interrupted: true })
    })

    // Chained turn: backend launches turn 2, no new user submit.
    event('message.start', 200)
    event('message.delta', 201, { text: 'chained reply' })
    event('message.complete', 202, { text: 'chained reply' })

    const state = stream.state(SID)
    expect((state.messages ?? []).some(m => m.role === 'assistant' && JSON.stringify(m.parts).includes('chained reply'))).toBe(true)
  })

  it('drops the cancelled turn\'s late terminal frame instead of claiming the live turn', () => {
    event('message.start', 100)
    event('message.delta', 101, { text: 'cancelled partial' })

    // Stop pressed, then a chained turn starts — message.start clears the Stop
    // latch, so the cancelled turn's own terminal frame arrives with the latch
    // down while the chained turn is live and has streamed nothing yet.
    act(() => {
      const s = stream.states.get(SID)!
      stream.states.set(SID, { ...s, busy: false, awaitingResponse: false, streamId: null, interrupted: true })
    })

    event('message.start', 200)
    event('message.delta', 201, { text: 'live reply' })
    event('message.complete', 202, { status: 'interrupted', text: 'cancelled partial' })

    const state = stream.state(SID)
    // The stale frame settles nothing: the chained turn is still the live one.
    expect(state.turnLive).toBe(true)
    expect(state.busy).toBe(true)
    // And it never paints the cancelled turn's partial as that turn's answer.
    expect((state.messages ?? []).some(m => JSON.stringify(m.parts).includes('cancelled partial'))).toBe(false)
  })

  it('still settles a chained turn the user cancels itself', () => {
    event('message.start', 100)
    event('message.delta', 101, { text: 'cancelled partial' })

    act(() => {
      const s = stream.states.get(SID)!
      stream.states.set(SID, { ...s, busy: false, awaitingResponse: false, streamId: null, interrupted: true })
    })

    event('message.start', 200)
    event('message.delta', 201, { text: 'live reply' })

    // The user stops the chained turn: cancelRun re-arms the latch, so its own
    // interrupted frame takes the normal settle path.
    act(() => {
      const s = stream.states.get(SID)!
      stream.states.set(SID, { ...s, busy: false, awaitingResponse: false, streamId: null, interrupted: true })
    })

    event('message.complete', 202, { status: 'interrupted', text: 'live reply' })

    const state = stream.state(SID)
    expect(state.busy).toBe(false)
    expect(state.turnLive).toBe(false)
    expect((state.messages ?? []).some(m => JSON.stringify(m.parts).includes('live reply'))).toBe(true)
  })
})
