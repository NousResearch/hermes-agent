import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'

import {
  $stalledSessionIds,
  clearAllSessionStates,
  LIVE_TURN_EVENT_SILENCE_MS,
  noteSessionEvent,
  publishSessionState,
  SESSION_WATCHDOG_TIMEOUT_MS
} from './session-states'

const busy = (mark: number) => {
  const state = createClientSessionState('stored-watchdog')

  state.busy = true
  state.turnStartedAt = mark

  return state
}

describe('session watchdog timer', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    clearAllSessionStates()
    $stalledSessionIds.set([])
  })

  afterEach(() => {
    clearAllSessionStates()
    $stalledSessionIds.set([])
    vi.useRealTimers()
  })

  it('re-arms one deadline per busy episode instead of swapping a timer per publish', () => {
    publishSessionState('rt-watchdog', busy(1))
    const timersAfterFirstPublish = vi.getTimerCount()

    for (let minute = 2; minute <= 12; minute += 1) {
      publishSessionState('rt-watchdog', busy(minute))
      expect(vi.getTimerCount()).toBe(timersAfterFirstPublish)
      vi.advanceTimersByTime(60_000)
    }

    // Continuous activity never looks stalled.
    expect($stalledSessionIds.get()).toEqual([])
  })

  it('allocates one silence timeout for a burst of events', () => {
    publishSessionState('rt-watchdog', busy(1))
    const timers = vi.spyOn(globalThis, 'setTimeout')

    for (let event = 0; event < 5; event += 1) {
      noteSessionEvent('rt-watchdog')
    }

    expect(timers.mock.calls.filter(([, delay]) => delay === LIVE_TURN_EVENT_SILENCE_MS)).toHaveLength(1)
  })

  it('allocates one watchdog timeout for a burst of busy publishes', () => {
    const timers = vi.spyOn(globalThis, 'setTimeout')

    for (let mark = 1; mark <= 5; mark += 1) {
      publishSessionState('rt-watchdog', busy(mark))
      vi.advanceTimersByTime(10_000)
    }

    expect(timers.mock.calls.filter(([, delay]) => delay === SESSION_WATCHDOG_TIMEOUT_MS)).toHaveLength(1)
  })

  it('still marks a busy session stalled after a full silent window', () => {
    publishSessionState('rt-watchdog', busy(1))

    vi.advanceTimersByTime(SESSION_WATCHDOG_TIMEOUT_MS + 1)

    expect($stalledSessionIds.get()).toContain('stored-watchdog')
  })
})
