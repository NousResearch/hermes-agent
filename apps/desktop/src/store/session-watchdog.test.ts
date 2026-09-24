import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'

import { $activeSessionId, $selectedStoredSessionId, $unreadFinishedSessionIds } from './session'
import {
  $stalledSessionIds,
  $workingSessionIds,
  clearAllSessionStates,
  publishSessionState,
  SESSION_WATCHDOG_TIMEOUT_MS
} from './session-states'

// Read from the store rather than restated here: these assert what happens on
// either side of the threshold, not what the threshold is.
const WATCHDOG_MS = SESSION_WATCHDOG_TIMEOUT_MS

function state(over: Partial<ClientSessionState> = {}): ClientSessionState {
  return { ...createClientSessionState(null), storedSessionId: 's1', ...over }
}

describe('session watchdog', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    clearAllSessionStates()
    $unreadFinishedSessionIds.set([])
    $selectedStoredSessionId.set(null)
    $activeSessionId.set(null)
  })

  afterEach(() => {
    vi.runOnlyPendingTimers()
    vi.useRealTimers()
    clearAllSessionStates()
    $unreadFinishedSessionIds.set([])
    $selectedStoredSessionId.set(null)
    $activeSessionId.set(null)
  })

  it('marks a silent session stalled without pretending it finished', () => {
    publishSessionState('rt1', state({ busy: true, storedSessionId: 's1' }))

    vi.advanceTimersByTime(WATCHDOG_MS)

    expect($workingSessionIds.get()).toContain('s1')
    expect($stalledSessionIds.get()).toContain('s1')
  })

  it('clears stalled on new activity and rearms the watchdog', () => {
    const working = state({ busy: true, storedSessionId: 's2' })
    publishSessionState('rt2', working)
    vi.advanceTimersByTime(WATCHDOG_MS)
    expect($stalledSessionIds.get()).toContain('s2')

    publishSessionState('rt2', { ...working, awaitingResponse: true })
    expect($stalledSessionIds.get()).not.toContain('s2')

    vi.advanceTimersByTime(WATCHDOG_MS - 1)
    expect($stalledSessionIds.get()).not.toContain('s2')
    expect($workingSessionIds.get()).toContain('s2')
  })

  it('clears both running and stalled on an authoritative terminal transition', () => {
    const working = state({ busy: true, storedSessionId: 's3' })
    publishSessionState('rt3', working)
    vi.advanceTimersByTime(WATCHDOG_MS)
    expect($stalledSessionIds.get()).toContain('s3')

    publishSessionState('rt3', { ...working, busy: false })

    expect($workingSessionIds.get()).not.toContain('s3')
    expect($stalledSessionIds.get()).not.toContain('s3')
  })

  it('never marks a session stalled when it settles before the window', () => {
    const working = state({ busy: true, storedSessionId: 's4' })
    publishSessionState('rt4', working)
    publishSessionState('rt4', { ...working, busy: false })
    vi.advanceTimersByTime(WATCHDOG_MS)

    expect($workingSessionIds.get()).not.toContain('s4')
    expect($stalledSessionIds.get()).not.toContain('s4')
  })

  it('clears stalled state and disarms timers on a gateway wipe', () => {
    publishSessionState('rt1', state({ busy: true, storedSessionId: 's1' }))
    vi.advanceTimersByTime(WATCHDOG_MS)
    expect($stalledSessionIds.get()).toEqual(['s1'])

    clearAllSessionStates()
    vi.advanceTimersByTime(WATCHDOG_MS)

    expect($workingSessionIds.get()).toEqual([])
    expect($stalledSessionIds.get()).toEqual([])
  })
})

describe('computed $workingSessionIds', () => {
  beforeEach(() => {
    clearAllSessionStates()
  })

  afterEach(() => {
    clearAllSessionStates()
  })

  it('reflects busy sessions under the id their surfaces key on', () => {
    publishSessionState('rt1', state({ busy: true, storedSessionId: 's1' }))
    publishSessionState('rt2', state({ busy: false, storedSessionId: 's2' }))
    // Not yet persisted, so the runtime id is the only id it has — and the one
    // the row is keyed by until the backend hands a stored id back.
    publishSessionState('rt3', state({ busy: true, storedSessionId: null }))

    expect($workingSessionIds.get()).toEqual(['s1', 'rt3'])
  })

  it('updates when session state changes', () => {
    publishSessionState('rt1', state({ busy: true, storedSessionId: 's1' }))
    expect($workingSessionIds.get()).toEqual(['s1'])

    publishSessionState('rt1', state({ busy: false, storedSessionId: 's1' }))
    expect($workingSessionIds.get()).toEqual([])
  })
})
