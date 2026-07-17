import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'

import { $activeSessionId, $selectedStoredSessionId, $unreadFinishedSessionIds } from './session'
import {
  $attentionSessionIds,
  $sessionStates,
  $workingSessionIds,
  clearAllSessionStates,
  getRecentlySettledSessionIds,
  publishSessionState,
  setWatchdogClearFn
} from './session-states'

const WATCHDOG_MS = 8 * 60 * 1000

function state(over: Partial<ClientSessionState> = {}): ClientSessionState {
  return { ...createClientSessionState(null), storedSessionId: 's1', ...over }
}

describe('session status transitions', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.setSystemTime(0)
    // clearAllSessionStates also disarms watchdog timers + drops settle-grace
    // entries, so no leftover state can leak in from a previous test.
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

    // idle → working
    const next = { ...s, busy: true }
    publishSessionState('rt1', next)

    expect($workingSessionIds.get()).toContain('s1')
  })

  it('removes a session from $workingSessionIds when busy transitions to false', () => {
    const s = state({ busy: true, storedSessionId: 's1' })
    publishSessionState('rt1', s)
    // Simulate the working state being set
    const working = { ...s, busy: true }
    publishSessionState('rt1', working)

    expect($workingSessionIds.get()).toContain('s1')

    // Now transition to idle
    const idle = { ...working, busy: false }
    publishSessionState('rt1', idle)

    expect($workingSessionIds.get()).not.toContain('s1')
  })

  it('adds a session to $attentionSessionIds when needsInput is true', () => {
    const s = state({ busy: true, needsInput: false, storedSessionId: 's1' })
    publishSessionState('rt1', s)

    const next = { ...s, needsInput: true }
    publishSessionState('rt1', next)

    expect($attentionSessionIds.get()).toContain('s1')
  })

  it('marks a background session unread when its turn finishes', () => {
    $selectedStoredSessionId.set('other-session')

    const working = state({ busy: true, storedSessionId: 's1' })
    publishSessionState('rt1', working)

    const idle = { ...working, busy: false }
    publishSessionState('rt1', idle)

    expect($unreadFinishedSessionIds.get()).toEqual(['s1'])
  })

  it('does NOT mark unread when the finishing session is the active one', () => {
    $selectedStoredSessionId.set('s1')

    const working = state({ busy: true, storedSessionId: 's1' })
    publishSessionState('rt1', working)

    const idle = { ...working, busy: false }
    publishSessionState('rt1', idle)

    expect($unreadFinishedSessionIds.get()).toEqual([])
  })

  it('does NOT mark unread on idle→idle re-asserts (no prior working state)', () => {
    $selectedStoredSessionId.set('other-session')

    const idle = state({ busy: false, storedSessionId: 's1' })
    publishSessionState('rt1', idle)

    expect($unreadFinishedSessionIds.get()).toEqual([])
  })

  it('grants settle grace when a working session goes idle', () => {
    $selectedStoredSessionId.set('other')

    const working = state({ busy: true, storedSessionId: 's1' })
    publishSessionState('rt1', working)

    const idle = { ...working, busy: false }
    publishSessionState('rt1', idle)

    expect(getRecentlySettledSessionIds()).toEqual(['s1'])
  })

  it('does not grant grace on idle→idle re-asserts', () => {
    const idle = state({ busy: false, storedSessionId: 's1' })
    publishSessionState('rt1', idle)
    expect(getRecentlySettledSessionIds()).toEqual([])
  })

  it('clears settle grace when the session goes busy again', () => {
    $selectedStoredSessionId.set('other')

    const working = state({ busy: true, storedSessionId: 's2' })
    publishSessionState('rt1', working)

    const idle = { ...working, busy: false }
    publishSessionState('rt1', idle)

    expect(getRecentlySettledSessionIds()).toEqual(['s2'])

    // New turn for the same session
    const workingAgain = { ...idle, busy: true }
    publishSessionState('rt1', workingAgain)

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

  it('drops a stuck session from $workingSessionIds once the silence window elapses', () => {
    // Wire a clear fn like use-session-state-cache does in the real app: the
    // watchdog hands us the runtime id, we publish the busy:false state.
    const clearedRuntimeIds: string[] = []
    setWatchdogClearFn(runtimeId => {
      clearedRuntimeIds.push(runtimeId)
      const current = $sessionStates.get()[runtimeId]

      if (current) {
        publishSessionState(runtimeId, { ...current, busy: false, needsInput: false })
      }
    })

    const working = state({ busy: true, storedSessionId: 's1' })
    publishSessionState('rt1', working)

    expect($workingSessionIds.get()).toContain('s1')

    // Watchdog fires after 8 min of silence → the wired clear fn runs and the
    // computed working set drops the session. This asserts the timer→callback
    // wiring, not just the projection.
    vi.advanceTimersByTime(WATCHDOG_MS)

    expect(clearedRuntimeIds).toEqual(['rt1'])
    expect($workingSessionIds.get()).not.toContain('s1')

    setWatchdogClearFn(null)
  })

  it('never fires for a session that settles before the window', () => {
    const clearedRuntimeIds: string[] = []
    setWatchdogClearFn(runtimeId => clearedRuntimeIds.push(runtimeId))

    const working = state({ busy: true, storedSessionId: 's2' })
    publishSessionState('rt2', working)

    // Session settles before the watchdog window
    const idle = { ...working, busy: false }
    publishSessionState('rt2', idle)

    vi.advanceTimersByTime(WATCHDOG_MS)

    // The watchdog was disarmed — the clear fn never ran.
    expect(clearedRuntimeIds).toEqual([])
    expect($workingSessionIds.get()).not.toContain('s2')

    setWatchdogClearFn(null)
  })

  it('does not fire after clearAllSessionStates disarms every timer', () => {
    const clearedRuntimeIds: string[] = []
    setWatchdogClearFn(runtimeId => clearedRuntimeIds.push(runtimeId))

    publishSessionState('rt1', state({ busy: true, storedSessionId: 's1' }))
    clearAllSessionStates()

    vi.advanceTimersByTime(WATCHDOG_MS)

    expect(clearedRuntimeIds).toEqual([])

    setWatchdogClearFn(null)
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
