import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $selectedStoredSessionId, $unreadFinishedSessionIds } from '@/store/session'
import { $attentionSessionIds, $workingSessionIds, clearAllSessionStates } from '@/store/session-states'

import { rehydrateLiveSessionStatuses } from './use-background-sync'

/**
 * `session.active_list` is the authoritative snapshot of what is RUNNING in the
 * polled gateway process. A session that finished while Desktop was looking
 * elsewhere — or whose runtime id was recycled by a backend respawn — simply
 * stops appearing in the response. Absence is therefore a completion signal,
 * not "no news": if nothing reaps it, the row spins forever and the
 * busy→idle edge that paints the green "your turn" dot never fires.
 */
describe('rehydrateLiveSessionStatuses — reaping vanished runtimes', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    $selectedStoredSessionId.set(null)
    $unreadFinishedSessionIds.set([])
  })

  afterEach(() => {
    vi.clearAllTimers()
    vi.useRealTimers()
    clearAllSessionStates()
    $unreadFinishedSessionIds.set([])
  })

  it('clears a working session that disappears from the live snapshot', () => {
    rehydrateLiveSessionStatuses({
      sessions: [{ id: 'runtime-a', session_key: 'stored-a', status: 'working' }]
    })

    expect($workingSessionIds.get()).toEqual(['stored-a'])

    // The turn finished and the gateway reaped the session between polls.
    rehydrateLiveSessionStatuses({ sessions: [] })

    expect($workingSessionIds.get()).toEqual([])
  })

  it('fires the unread "your turn" marker for a vanished background session', () => {
    rehydrateLiveSessionStatuses({
      sessions: [{ id: 'runtime-b', session_key: 'stored-b', status: 'working' }]
    })

    rehydrateLiveSessionStatuses({ sessions: [] })

    expect($unreadFinishedSessionIds.get()).toEqual(['stored-b'])
  })

  it('clears a blocked session that disappears from the live snapshot', () => {
    rehydrateLiveSessionStatuses({
      sessions: [{ id: 'runtime-c', session_key: 'stored-c', status: 'waiting' }]
    })

    expect($attentionSessionIds.get()).toEqual(['stored-c'])

    rehydrateLiveSessionStatuses({ sessions: [] })

    expect($attentionSessionIds.get()).toEqual([])
  })

  it('leaves runtimes this poll never seeded alone', () => {
    // A background PROFILE's sessions are served by a different gateway and
    // never appear in this profile's active_list. Reaping them would dark out
    // every other profile's running rows.
    rehydrateLiveSessionStatuses(
      { sessions: [{ id: 'runtime-other', session_key: 'stored-other', status: 'working' }] },
      Date.now(),
      'other'
    )

    rehydrateLiveSessionStatuses({ sessions: [] }, Date.now(), 'default')

    expect($workingSessionIds.get()).toEqual(['stored-other'])
  })
})
