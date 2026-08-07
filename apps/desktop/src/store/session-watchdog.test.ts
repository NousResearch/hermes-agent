import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'

import { $activeSessionId, $selectedStoredSessionId, $unreadFinishedSessionIds } from './session'
import {
  $attentionSessionIds,
  $stalledSessionIds,
  $workingSessionIds,
  clearAllSessionStates,
  getRecentlySettledSessionIds,
  publishSessionState
} from './session-states'

const WATCHDOG_MS = 8 * 60 * 1000

function state(over: Partial<ClientSessionState> = {}): ClientSessionState {
  return { ...createClientSessionState(null), storedSessionId: 's1', ...over }
}

describe('session status transitions', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.setSystemTime(0)
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

  it('adds a session to $workingSessionIds when busy transitions to true', () => {
    const s = state({ busy: false, storedSessionId: 's1' })
    publishSessionState('rt1', s)

    publishSessionState('rt1', { ...s, busy: true })

    expect($workingSessionIds.get()).toContain('s1')
  })

  it('removes a session from $workingSessionIds when busy transitions to false', () => {
    const working = state({ busy: true, storedSessionId: 's1' })
    publishSessionState('rt1', working)

    expect($workingSessionIds.get()).toContain('s1')

    publishSessionState('rt1', { ...working, busy: false })

    expect($workingSessionIds.get()).not.toContain('s1')
  })

  it('adds a session to $attentionSessionIds when needsInput is true', () => {
    const s = state({ busy: true, needsInput: false, storedSessionId: 's1' })
    publishSessionState('rt1', s)

    publishSessionState('rt1', { ...s, needsInput: true })

    expect($attentionSessionIds.get()).toContain('s1')
  })

  it('marks a background session unread when its turn finishes', () => {
    $selectedStoredSessionId.set('other-session')
    const working = state({ busy: true, storedSessionId: 's1' })
    publishSessionState('rt1', working)

    publishSessionState('rt1', { ...working, busy: false })

    expect($unreadFinishedSessionIds.get()).toEqual(['s1'])
  })

  it('does NOT mark unread when the finishing session is the active one', () => {
    $selectedStoredSessionId.set('s1')
    const working = state({ busy: true, storedSessionId: 's1' })
    publishSessionState('rt1', working)

    publishSessionState('rt1', { ...working, busy: false })

    expect($unreadFinishedSessionIds.get()).toEqual([])
  })

  it('does NOT mark unread on idle→idle re-asserts (no prior working state)', () => {
    $selectedStoredSessionId.set('other-session')
    publishSessionState('rt1', state({ busy: false, storedSessionId: 's1' }))

    expect($unreadFinishedSessionIds.get()).toEqual([])
  })

  it('grants settle grace when a working session goes idle', () => {
    $selectedStoredSessionId.set('other')
    const working = state({ busy: true, storedSessionId: 's1' })
    publishSessionState('rt1', working)

    publishSessionState('rt1', { ...working, busy: false })

    expect(getRecentlySettledSessionIds()).toEqual(['s1'])
  })

  it('does not grant grace on idle→idle re-asserts', () => {
    publishSessionState('rt1', state({ busy: false, storedSessionId: 's1' }))
    expect(getRecentlySettledSessionIds()).toEqual([])
  })

  it('clears settle grace when the session goes busy again', () => {
    $selectedStoredSessionId.set('other')
    const working = state({ busy: true, storedSessionId: 's2' })
    publishSessionState('rt1', working)
    const idle = { ...working, busy: false }
    publishSessionState('rt1', idle)

    expect(getRecentlySettledSessionIds()).toEqual(['s2'])

    publishSessionState('rt1', { ...idle, busy: true })

    expect(getRecentlySettledSessionIds()).toEqual([])
  })
})

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

  it('is empty when no sessions are busy', () => {
    expect($workingSessionIds.get()).toEqual([])
  })

  it('reflects sessions with busy=true and a storedSessionId', () => {
    publishSessionState('rt1', state({ busy: true, storedSessionId: 's1' }))
    publishSessionState('rt2', state({ busy: false, storedSessionId: 's2' }))
    publishSessionState('rt3', state({ busy: true, storedSessionId: null }))

    expect($workingSessionIds.get()).toEqual(['s1'])
  })

  it('updates when session state changes', () => {
    publishSessionState('rt1', state({ busy: true, storedSessionId: 's1' }))
    expect($workingSessionIds.get()).toEqual(['s1'])

    publishSessionState('rt1', state({ busy: false, storedSessionId: 's1' }))
    expect($workingSessionIds.get()).toEqual([])
  })
})

describe('computed $attentionSessionIds', () => {
  beforeEach(() => {
    clearAllSessionStates()
  })

  afterEach(() => {
    clearAllSessionStates()
  })

  it('reflects sessions with needsInput=true and a storedSessionId', () => {
    publishSessionState('rt1', state({ needsInput: true, storedSessionId: 's1' }))
    publishSessionState('rt2', state({ needsInput: false, storedSessionId: 's2' }))

    expect($attentionSessionIds.get()).toEqual(['s1'])
  })

  it('clears when $sessionStates is cleared', () => {
    publishSessionState('rt1', state({ needsInput: true, storedSessionId: 's1' }))
    expect($attentionSessionIds.get()).toEqual(['s1'])

    clearAllSessionStates()
    expect($attentionSessionIds.get()).toEqual([])
  })
})
