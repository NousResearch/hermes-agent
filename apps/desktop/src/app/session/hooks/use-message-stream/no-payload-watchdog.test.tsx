// A turn that went live (message.start) but whose backend died before producing
// ANY assistant payload leaves turnLive=busy=true forever — no message.complete
// arrives, and a dead gateway stops heartbeating, so the existing session.info
// running=false settle never fires. The 5-min session watchdog is the only
// existing backstop; the no-payload watchdog bounds that wait to 60s.
//
// The watchdog is lazy-armed: a bounded timer exists only while a session sits
// in the stuck shape (turnLive, no payload, busy, awaitingResponse, clock
// seeded), armed from the updateSessionState chokepoint. No global interval —
// that would break consumers that count outstanding timers.
import { act, cleanup } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { RpcEvent } from '@/types/hermes'

import { type MessageStreamHarness, renderMessageStream } from './test-harness'
import { NO_PAYLOAD_WATCHDOG_MS } from './index'

const SID = 'no-payload-watchdog-session'

let stream: MessageStreamHarness

async function mountHarness() {
  vi.useFakeTimers()
  stream = renderMessageStream(SID)
  await act(async () => {
    await Promise.resolve()
  })
}

describe('no-payload watchdog settles a live turn that produced nothing', () => {
  afterEach(() => {
    cleanup()
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it('force-settles a live turn with no payload once the watchdog deadline passes', async () => {
    await mountHarness()

    // message.start goes live and arms the watchdog (busy, no payload yet).
    emit({ session_id: SID, type: 'message.start', payload: {} })

    // The backend dies: no deltas, no message.complete, no heartbeats.
    // The fire→settle can rearm once (fire lands a tick after the deadline),
    // so advance well past it in bounded steps until settled.
    for (let i = 0; i < 5 && stream.state(SID).busy; i += 1) {
      await act(async () => {
        await vi.advanceTimersByTimeAsync(NO_PAYLOAD_WATCHDOG_MS / 2)
      })
    }

    const state = stream.state(SID)
    expect(state.busy).toBe(false)
    expect(state.awaitingResponse).toBe(false)
    expect(state.turnLive).toBe(false)
    expect(state.turnStartedAt).toBeNull()
    expect(state.streamId).toBeNull()
    // The empty placeholder bubble is not stranded pending forever.
    expect(state.messages.every(message => !message.pending)).toBe(true)
  })

  it('does NOT settle a live turn whose payload arrived', async () => {
    await mountHarness()

    emit({ session_id: SID, type: 'message.start', payload: {} })
    emit({ payload: { text: 'partial answer' }, session_id: SID, type: 'message.delta' })

    await act(async () => {
      await vi.advanceTimersByTimeAsync(NO_PAYLOAD_WATCHDOG_MS + 1_000)
    })

    // Payload arrived — the turn is legitimately still working; the watchdog
    // must leave it alone (message.complete / session.info will settle it).
    const state = stream.state(SID)
    expect(state.busy).toBe(true)
  })

  it('does NOT settle a healthy turn mid-flight before the deadline', async () => {
    await mountHarness()

    emit({ session_id: SID, type: 'message.start', payload: {} })

    await act(async () => {
      await vi.advanceTimersByTimeAsync(NO_PAYLOAD_WATCHDOG_MS - 5_000)
    })

    const state = stream.state(SID)
    expect(state.busy).toBe(true)
    expect(state.turnLive).toBe(true)
  })

  it('releases the watchdog timer once a turn settles normally', async () => {
    await mountHarness()

    emit({ session_id: SID, type: 'message.start', payload: {} })
    emit({ payload: { text: 'done' }, session_id: SID, type: 'message.complete' })
    // Advance well past where a stale watchdog would fire.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(NO_PAYLOAD_WATCHDOG_MS + 1_000)
    })

    const state = stream.state(SID)
    expect(state.busy).toBe(false)
    expect(state.messages.every(message => !message.pending)).toBe(true)
  })
})

const emit = (event: RpcEvent) => act(() => stream.handleEvent(event))
