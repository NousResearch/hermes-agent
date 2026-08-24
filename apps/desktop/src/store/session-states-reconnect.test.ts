import { registryBackendScopeKey } from '@hermes/shared'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'

import { $activeSessionId, $selectedStoredSessionId, $unreadFinishedSessionIds } from './session'
import {
  $attentionSessionIds,
  $stalledSessionIds,
  $workingSessionIds,
  clearAllSessionStates,
  publishSessionState,
  reconcileBusyStatesOnReconnect,
  recordSessionEventScope,
  SESSION_WATCHDOG_TIMEOUT_MS
} from './session-states'

function state(over: Partial<ClientSessionState> = {}): ClientSessionState {
  return { ...createClientSessionState(null), storedSessionId: 's1', ...over }
}

// The stale-flag half of #53902/#73082: a backend respawn re-mints runtime
// ids, so a pre-reconnect busy state never receives its terminal busy:false
// and the session's running arc stays armed forever. The reconnect paths call
// reconcileBusyStatesOnReconnect to retire those claims.
describe('reconcileBusyStatesOnReconnect', () => {
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

  it('clears a stale busy session on primary reconnect', () => {
    publishSessionState('rt1', state({ busy: true, storedSessionId: 's1' }))
    expect($workingSessionIds.get()).toContain('s1')

    reconcileBusyStatesOnReconnect()

    expect($workingSessionIds.get()).not.toContain('s1')
  })

  it('disarms the stall watchdog with the busy claim', () => {
    publishSessionState('rt1', state({ busy: true, storedSessionId: 's1' }))

    reconcileBusyStatesOnReconnect()

    // Without reconcile the watchdog would fire and paint s1 stalled.
    vi.advanceTimersByTime(SESSION_WATCHDOG_TIMEOUT_MS + 1000)
    expect($stalledSessionIds.get()).not.toContain('s1')
  })

  it('preserves needsInput — a blocking prompt is not a stale flag', () => {
    publishSessionState('rt1', state({ busy: true, needsInput: true, storedSessionId: 's1' }))
    expect($attentionSessionIds.get()).toContain('s1')

    reconcileBusyStatesOnReconnect()

    expect($workingSessionIds.get()).not.toContain('s1')
    expect($attentionSessionIds.get()).toContain('s1')
  })

  it('primary reconcile leaves registry-scoped sessions alone', () => {
    const scope = registryBackendScopeKey('connA', 'default')
    publishSessionState('rtA', state({ busy: true, storedSessionId: 'sA' }))
    recordSessionEventScope({ connectionId: 'connA', profile: 'default', session_id: 'rtA' })
    publishSessionState('rtLocal', state({ busy: true, storedSessionId: 'sLocal' }))

    reconcileBusyStatesOnReconnect()

    expect($workingSessionIds.get()).toContain('sA')
    expect($workingSessionIds.get()).not.toContain('sLocal')

    // And the scoped variant clears ONLY its own connection's sessions.
    reconcileBusyStatesOnReconnect(scope)
    expect($workingSessionIds.get()).not.toContain('sA')
  })

  it('scoped reconcile does not touch other connections or the primary', () => {
    publishSessionState('rtA', state({ busy: true, storedSessionId: 'sA' }))
    recordSessionEventScope({ connectionId: 'connA', profile: 'default', session_id: 'rtA' })
    publishSessionState('rtB', state({ busy: true, storedSessionId: 'sB' }))
    recordSessionEventScope({ connectionId: 'connB', profile: 'default', session_id: 'rtB' })
    publishSessionState('rtLocal', state({ busy: true, storedSessionId: 'sLocal' }))

    reconcileBusyStatesOnReconnect(registryBackendScopeKey('connA', 'default'))

    expect($workingSessionIds.get()).not.toContain('sA')
    expect($workingSessionIds.get()).toContain('sB')
    expect($workingSessionIds.get()).toContain('sLocal')
  })

  it('a live turn re-asserting busy after reconcile re-arms the arc', () => {
    const s = state({ busy: true, storedSessionId: 's1' })
    publishSessionState('rt1', s)
    reconcileBusyStatesOnReconnect()
    expect($workingSessionIds.get()).not.toContain('s1')

    // The still-alive backend's next event republishes busy under a live id.
    publishSessionState('rt2', state({ busy: true, storedSessionId: 's1' }))

    expect($workingSessionIds.get()).toContain('s1')
  })
})
