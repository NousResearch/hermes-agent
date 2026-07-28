import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $selectedStoredSessionId, $unreadFinishedSessionIds } from '@/store/session'
import { $workingSessionIds, clearAllSessionStates } from '@/store/session-states'

import { rehydrateLiveSessionStatuses, resetLiveRuntimeTracking } from './use-background-sync'

/**
 * (C) The sidebar spinner is driven by `$workingSessionIds`, which is keyed by
 * STORED session id. A turn that STARTS while Desktop isn't receiving stream
 * events — a background profile, a degraded remote socket, a session opened on
 * another surface — is only ever learned about through the `session.active_list`
 * poll. If that poll can't seed a row the renderer has never seen, the thread
 * name never gets its arc even though the backend is plainly working.
 */
describe('rehydrateLiveSessionStatuses — seeding a turn the renderer never saw start', () => {
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
  })

  it('shows the spinner for a turn that started with no stream events', () => {
    rehydrateLiveSessionStatuses({
      sessions: [{ id: 'runtime-cold', session_key: 'stored-cold', status: 'working' }]
    })

    expect($workingSessionIds.get()).toContain('stored-cold')
  })

  it('keeps the spinner across polls while the turn is still running', () => {
    rehydrateLiveSessionStatuses({
      sessions: [{ id: 'runtime-cold', session_key: 'stored-cold', status: 'working' }]
    })
    rehydrateLiveSessionStatuses({
      sessions: [{ id: 'runtime-cold', session_key: 'stored-cold', status: 'working' }]
    })

    expect($workingSessionIds.get()).toContain('stored-cold')
  })

  it('shows the spinner when a runtime id is recycled onto a new stored session', () => {
    // A respawned backend can mint the same runtime id for a different stored
    // session. The row for the NEW stored id must light up, not the stale one.
    rehydrateLiveSessionStatuses({
      sessions: [{ id: 'runtime-1', session_key: 'stored-old', status: 'working' }]
    })

    rehydrateLiveSessionStatuses({
      sessions: [{ id: 'runtime-1', session_key: 'stored-new', status: 'working' }]
    })

    expect($workingSessionIds.get()).toContain('stored-new')
    expect($workingSessionIds.get()).not.toContain('stored-old')
  })

  it('leaves a starting session idle — the agent build is not proof of a turn', () => {
    // `starting` = `agent_build_started` without `agent_ready`. _start_agent_build
    // runs on the first prompt OR any incidental RPC that needs the agent, so it
    // is not proof of a turn — lighting the spinner here would fire on merely
    // opening a session. A real turn arrives as `working`.
    rehydrateLiveSessionStatuses({
      sessions: [{ id: 'runtime-boot', session_key: 'stored-boot', status: 'starting' }]
    })

    expect($workingSessionIds.get()).not.toContain('stored-boot')
  })
})
