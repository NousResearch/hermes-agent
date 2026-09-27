import type { GatewayEvent, GatewayEventName } from '@hermes/shared'
import { act, cleanup } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $currentUsage } from '@/store/session'

import { type MessageStreamHarness, renderMessageStream } from './test-harness'

const SID = 'session-1'
// $currentUsage mirrors the primary session; ClientSessionState.usage drives
// the same status bar when a secondary tile is focused.
const BASELINE = { calls: 2, input: 500, output: 40, total: 540 }

let stream: MessageStreamHarness
let sessionStates = new Map<string, ClientSessionState>()

function mountStream() {
  stream = renderMessageStream(SID, { states: sessionStates })
}

describe('useMessageStream status-bar usage scoping', () => {
  beforeEach(() => {
    sessionStates = new Map([[SID, { ...createClientSessionState(), usage: { ...BASELINE } }]])
    $currentUsage.set({ ...BASELINE })
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('merges a live session.usage tick from the focused session', () => {
    mountStream()

    act(() =>
      stream.handleEvent({
        payload: { usage: { context_percent: 42, input: 1200, total: 1280 } },
        session_id: SID,
        type: 'session.usage'
      })
    )

    // Merge, not replace: fields absent from the tick keep their prior values.
    expect($currentUsage.get()).toEqual({ ...BASELINE, context_percent: 42, input: 1200, total: 1280 })
    expect(sessionStates.get(SID)?.usage).toEqual({
      ...BASELINE,
      context_percent: 42,
      input: 1200,
      total: 1280
    })
  })

  it('caches a background session.usage tick without overwriting the primary status bar', () => {
    mountStream()

    act(() =>
      stream.handleEvent({
        payload: { usage: { input: 9999, total: 9999 } },
        session_id: 'background-session',
        type: 'session.usage'
      })
    )

    expect($currentUsage.get()).toEqual(BASELINE)
    expect(sessionStates.get('background-session')?.usage).toEqual({
      calls: 0,
      input: 9999,
      output: 0,
      total: 9999
    })
  })

  it('applies message.complete usage from the focused session', () => {
    mountStream()

    act(() =>
      stream.handleEvent({
        payload: { text: 'done', usage: { calls: 3, input: 1500, output: 90, total: 1590 } },
        session_id: SID,
        type: 'message.complete'
      })
    )

    expect($currentUsage.get()).toEqual({ calls: 3, input: 1500, output: 90, total: 1590 })
  })

  it('ignores message.complete usage from a background session', () => {
    mountStream()

    act(() =>
      stream.handleEvent({
        payload: { text: 'done', usage: { calls: 9, input: 9999, output: 999, total: 9999 } },
        session_id: 'background-session',
        type: 'message.complete'
      })
    )

    expect($currentUsage.get()).toEqual(BASELINE)
  })

  it('pushes stream_tps onto the status bar after >=3 streamed deltas, then clears it on complete', () => {
    mountStream()

    // Deltas carry gateway timestamps (unix SECONDS, converted to ms by the
    // reducer). First emit happens immediately, then throttled ~1s, until 3
    // samples sit in the 2s window: Math.round((3/2)*10)/10 = 1.5.
    const delta = (timestamp: number, text: string): GatewayEvent<'message.delta'> => ({
      payload: { text, timestamp } as GatewayEvent<'message.delta'>['payload'],
      session_id: SID,
      type: 'message.delta'
    })

    act(() => stream.handleEvent(delta(1.0, 'a')))
    act(() => stream.handleEvent(delta(1.1, 'b')))

    // 2 samples so far (1.1 throttled) — still no rate.
    expect($currentUsage.get().stream_tps).toBeUndefined()

    // 1+ s after the last emit: 3 samples in the 2s window => a rate.
    act(() => stream.handleEvent(delta(2.11, 'c')))
    expect($currentUsage.get().stream_tps).toBe(1.5)

    // End of turn: the live rate is cleared so it doesn't stay frozen.
    act(() => stream.handleEvent({ payload: { text: 'done' }, session_id: SID, type: 'message.complete' }))
    expect($currentUsage.get().stream_tps).toBeUndefined()
    // Other usage fields are untouched by the tps lifecycle.
    expect($currentUsage.get()).toEqual(BASELINE)
  })

  it('keeps $currentUsage free of stream_tps before enough deltas arrive', () => {
    mountStream()

    const delta = (timestamp: number, text: string): GatewayEvent<'message.delta'> => ({
      payload: { text, timestamp } as GatewayEvent<'message.delta'>['payload'],
      session_id: SID,
      type: 'message.delta'
    })

    // Only 2 deltas streamed and then the turn completes — under the 3-sample
    // minimum, so no rate was ever published and none lingers after the reset.
    act(() => stream.handleEvent(delta(1.0, 'a')))
    act(() => stream.handleEvent(delta(1.1, 'b')))
    act(() => stream.handleEvent({ payload: { text: 'done' }, session_id: SID, type: 'message.complete' }))

    expect($currentUsage.get()).toEqual(BASELINE)
  })

  it('does not publish a background session stream_tps onto the primary status bar', () => {
    mountStream()

    // Background session 'bg-1' streams enough to produce a rate, but the
    // reducer only publishes the ACTIVE session's rate to $currentUsage.
    const bgDelta = (timestamp: number): GatewayEvent<'message.delta'> => ({
      payload: { text: 'x', timestamp } as GatewayEvent<'message.delta'>['payload'],
      session_id: 'bg-1',
      type: 'message.delta'
    })

    act(() => stream.handleEvent(bgDelta(1.0)))
    act(() => stream.handleEvent(bgDelta(1.1)))
    act(() => stream.handleEvent(bgDelta(1.2)))
    act(() => stream.handleEvent(bgDelta(2.16)))

    expect($currentUsage.get()).toEqual(BASELINE)
  })

  it('covers the active session stream_tps only in the focused tile, scoped per session', () => {
    // Same behavior asserted through the delta path: the counter is per-session
    // so one session's streaming never leaks into another's published rate.
    mountStream()

    const mkDelta = (timestamp: number): GatewayEvent<'message.delta'> => ({
      payload: { text: 'x', timestamp } as GatewayEvent<'message.delta'>['payload'],
      session_id: SID,
      type: 'message.delta'
    })

    act(() => stream.handleEvent(mkDelta(1.0)))
    act(() => stream.handleEvent(mkDelta(1.1)))
    act(() => stream.handleEvent(mkDelta(2.11)))
    expect($currentUsage.get().stream_tps).toBe(1.5)

    // A background session's deltas don't override the active one.
    act(() =>
      stream.handleEvent({
        payload: { text: 'x', timestamp: 2.16 },
        session_id: 'bg-2',
        type: 'message.delta'
      })
    )
    expect($currentUsage.get().stream_tps).toBe(1.5)

    act(() => stream.handleEvent({ payload: { text: 'done' }, session_id: SID, type: 'message.complete' }))
    expect($currentUsage.get().stream_tps).toBeUndefined()
  })
})
