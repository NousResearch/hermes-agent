import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'
import { $selectedStoredSessionId, $unreadFinishedSessionIds } from '@/store/session'
import {
  $sessionStates,
  $workingSessionIds,
  anyLiveTurnAwaitingEvents,
  clearAllSessionStates,
  LIVE_TURN_EVENT_SILENCE_MS,
  noteSessionEvent,
  publishSessionState
} from '@/store/session-states'

import { rehydrateLiveSessionStatuses, resetLiveRuntimeTracking } from './use-background-sync'

/**
 * (#125306) A turn that produces nothing for LIVE_TURN_EVENT_SILENCE_MS is
 * force-settled as a stream drop. The `session.active_list` poll is that
 * watchdog's only witness for a turn the backend is still driving — a local
 * model prefills for minutes with no stream events and no state.db writes.
 * These pin what the poll report counts as "still working".
 */
describe('rehydrateLiveSessionStatuses — feeding the silence watchdog', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    $selectedStoredSessionId.set(null)
    $unreadFinishedSessionIds.set([])
    resetLiveRuntimeTracking()
  })

  afterEach(() => {
    vi.clearAllTimers()
    vi.useRealTimers()
    clearAllSessionStates()
    resetLiveRuntimeTracking()
    $unreadFinishedSessionIds.set([])
    $selectedStoredSessionId.set(null)
  })

  function liveTurn(runtimeId: string, storedId: string): void {
    publishSessionState(runtimeId, {
      ...createClientSessionState(storedId),
      busy: true,
      turnLive: true
    })
    // Arm the silence clock the way the gateway-event path does on the turn's
    // last attributed event.
    noteSessionEvent(runtimeId)
  }

  // `interrupted` is the force-settle stamp; the stream_drop card itself rides
  // the transcript, which an unreferenced runtime releases on settle.
  function forceSettled(runtimeId: string): boolean {
    return Boolean($sessionStates.get()[runtimeId]?.interrupted)
  }

  it("resets the silence clock on a 'starting' turn (agent still being built)", () => {
    liveTurn('runtime-build', 'stored-build')

    vi.advanceTimersByTime(LIVE_TURN_EVENT_SILENCE_MS - 1_000)
    rehydrateLiveSessionStatuses({
      sessions: [{ id: 'runtime-build', session_key: 'stored-build', status: 'starting' }]
    })
    vi.advanceTimersByTime(LIVE_TURN_EVENT_SILENCE_MS - 1_000)

    // 88s total without a stream event, but the poll kept vouching for the
    // turn: no force-settle, no stream_drop card. The spinner contract is
    // unchanged, though — a build alone is not proof of a turn.
    expect(forceSettled('runtime-build')).toBe(false)
    expect($workingSessionIds.get()).not.toContain('stored-build')
  })

  it("resets the silence clock on a 'working' turn past the backstop window", () => {
    liveTurn('runtime-slow', 'stored-slow')

    vi.advanceTimersByTime(LIVE_TURN_EVENT_SILENCE_MS - 1_000)
    rehydrateLiveSessionStatuses({
      sessions: [{ id: 'runtime-slow', session_key: 'stored-slow', status: 'working' }]
    })
    vi.advanceTimersByTime(LIVE_TURN_EVENT_SILENCE_MS - 1_000)

    expect(forceSettled('runtime-slow')).toBe(false)
  })

  it("does not vouch for an 'idle' row — a truly finished turn still settles", () => {
    liveTurn('runtime-idle', 'stored-idle')

    vi.advanceTimersByTime(LIVE_TURN_EVENT_SILENCE_MS - 1_000)
    rehydrateLiveSessionStatuses({
      sessions: [{ id: 'runtime-idle', session_key: 'stored-idle', status: 'idle' }]
    })
    vi.advanceTimersByTime(LIVE_TURN_EVENT_SILENCE_MS)

    expect(forceSettled('runtime-idle')).toBe(true)
  })
})

describe('anyLiveTurnAwaitingEvents', () => {
  afterEach(() => {
    clearAllSessionStates()
  })

  it('is true only while a session has a live turn that is not waiting on the user', () => {
    expect(anyLiveTurnAwaitingEvents()).toBe(false)

    publishSessionState('rt', { ...createClientSessionState('s'), busy: true, turnLive: true })
    expect(anyLiveTurnAwaitingEvents()).toBe(true)

    publishSessionState('rt', { ...createClientSessionState('s'), busy: false, turnLive: false })
    expect(anyLiveTurnAwaitingEvents()).toBe(false)

    publishSessionState('rt', { ...createClientSessionState('s'), busy: true, needsInput: true })
    expect(anyLiveTurnAwaitingEvents()).toBe(false)
  })
})
