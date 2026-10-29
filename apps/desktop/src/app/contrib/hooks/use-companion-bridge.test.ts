import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'
import { $companionActivity, $companionState, setCompanionActivity } from '@/store/companion'
import { clearAllSessionStates, publishSessionState } from '@/store/session-states'

import { syncCompanionBusyFromSessions } from './use-companion-bridge'

const busySession = (storedId: string) => ({ ...createClientSessionState(storedId), busy: true })

describe('syncCompanionBusyFromSessions', () => {
  beforeEach(() => {
    clearAllSessionStates()
    setCompanionActivity({ busy: false, toolRunning: false, reasoning: false, awaitingInput: false })
  })

  afterEach(() => {
    clearAllSessionStates()
    setCompanionActivity({ busy: false, toolRunning: false, reasoning: false, awaitingInput: false })
  })

  it('marks the companion busy while any session is working, foreground or not', () => {
    // A background session is running — its per-session stream hook is NOT
    // mounted, so nothing ever set toolRunning/reasoning on $companionActivity.
    publishSessionState('runtime-background', busySession('stored-background'))

    syncCompanionBusyFromSessions()

    expect($companionActivity.get().busy).toBe(true)
    // busy maps to the `run` pose even without a foreground toolRunning signal.
    expect($companionState.get()).toBe('run')
  })

  it('clears busy and returns to idle once every session is at rest', () => {
    publishSessionState('runtime-background', busySession('stored-background'))
    syncCompanionBusyFromSessions()
    expect($companionActivity.get().busy).toBe(true)

    // Turn ends: the session is no longer busy.
    publishSessionState('runtime-background', {
      ...createClientSessionState('stored-background'),
      busy: false
    })

    syncCompanionBusyFromSessions()

    expect($companionActivity.get().busy).toBe(false)
    expect($companionState.get()).toBe('idle')
  })

  it('stays busy when only one of several sessions finishes', () => {
    publishSessionState('runtime-a', busySession('stored-a'))
    publishSessionState('runtime-b', busySession('stored-b'))
    syncCompanionBusyFromSessions()
    expect($companionActivity.get().busy).toBe(true)

    // Session A finishes; B is still running.
    publishSessionState('runtime-a', { ...createClientSessionState('stored-a'), busy: false })
    syncCompanionBusyFromSessions()

    expect($companionActivity.get().busy).toBe(true)
    expect($companionState.get()).toBe('run')
  })
})
