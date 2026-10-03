import type { GatewayEvent } from '@hermes/shared'
import { act, cleanup } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { type MessageStreamHarness, renderMessageStream } from './test-harness'

const SID = 'interrupted-latch-session'

let stream: MessageStreamHarness

const event = (type: GatewayEvent['type'], payload: Record<string, unknown> = {}) =>
  act(() => {
    stream.handleEvent({ payload: { ...payload }, session_id: SID, type })
  })

describe('interrupted latch releases on the turn-exit confirmation (#122723)', () => {
  beforeEach(() => {
    stream = renderMessageStream(SID)
  })

  afterEach(() => {
    cleanup()
  })

  // cancelRun's client-side writes: busy=false, awaitingResponse=false,
  // streamId=null, interrupted=true (use-prompt-actions/index.ts stop path).
  const pressStop = () =>
    act(() => {
      const s = stream.states.get(SID)!
      stream.states.set(SID, {
        ...s,
        busy: false,
        awaitingResponse: false,
        streamId: null,
        interrupted: true
      })
    })

  it('arms and appends a turn whose message.start arrives after an interrupt was cleared', () => {
    // A fresh submit cleared the latch (seedOptimistic writes interrupted:false)
    // and armed the optimistic user row — the control the issue's reporter ran.
    event('message.start')
    event('message.delta', { text: 'fresh answer' })

    act(() => {
      const s = stream.states.get(SID)!
      stream.states.set(SID, { ...s, interrupted: false, busy: true, awaitingResponse: true })
    })

    event('message.start')
    event('message.delta', { text: 'fresh answer' })
    event('message.complete', { text: 'fresh answer' })

    const state = stream.state(SID)
    expect(state.busy).toBe(false)
    expect(
      (state.messages ?? []).some(
        m => m.role === 'assistant' && JSON.stringify(m.parts).includes('fresh answer')
      )
    ).toBe(true)
  })

  it('paints a backend-chained turn whose message.start races the interrupt latch', () => {
    // The repro shape: Stop pressed mid-turn; the backend launches its own
    // follow-up turn (goal continuation / completion drain) with NO new user
    // submit. The cancelled turn's exit emits the authoritative running=false
    // session.info BEFORE the chained turn's message.start — that confirmation
    // must release the latch so the chained turn paints.
    event('message.start')
    event('message.delta', { text: 'goal reply' })
    event('message.complete', { text: 'goal reply', status: 'interrupted' })

    pressStop()

    // The cancelled turn's exit: the agent loop's finally block settles the
    // session (running=false). This is the backend proving the interrupt landed.
    event('session.info', { running: false })

    // The chained turn arrives — no user submit precedes it.
    event('message.start')
    event('message.delta', { text: 'chained reply' })
    event('message.complete', { text: 'chained reply' })

    const state = stream.state(SID)
    expect(
      (state.messages ?? []).some(
        m => m.role === 'assistant' && JSON.stringify(m.parts).includes('chained reply')
      )
    ).toBe(true)
  })

  it('still drops the cancelled turn own stale deltas before the exit confirmation', () => {
    // The latch must hold until the backend confirms the exit: a late delta
    // from the cancelled turn (cooperative cancel not yet propagated) is
    // suppressed exactly as before.
    event('message.start')
    event('message.delta', { text: 'partial goal reply' })
    event('message.complete', { text: 'partial goal reply', status: 'interrupted' })

    pressStop()

    // Stale delta from the still-dying turn, BEFORE its exit confirmation.
    event('message.delta', { text: ' STALE TAIL' })

    expect(stream.text(SID)).not.toContain('STALE TAIL')

    // Exit confirmation arrives; only now does the latch release.
    event('session.info', { running: false })
    expect(stream.state(SID).interrupted).toBe(false)
  })
})
