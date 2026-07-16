import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { sessionIdentityKey } from '@/lib/session-identity'

import { $workingSessionIds, onSessionWatchdogClear, setSessionWorking, setWorkingSessionIds } from './session'

const WATCHDOG_MS = 8 * 60 * 1000

describe('session watchdog', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    setWorkingSessionIds(() => [])
  })

  afterEach(() => {
    vi.runOnlyPendingTimers()
    vi.useRealTimers()
  })

  it('drops a stuck session and notifies listeners once the silence window elapses', () => {
    const cleared: string[] = []
    const off = onSessionWatchdogClear(id => cleared.push(id))

    setSessionWorking('s1', true)
    expect($workingSessionIds.get()).toContain(sessionIdentityKey('s1', 'default'))

    vi.advanceTimersByTime(WATCHDOG_MS)

    // Both the sidebar dot AND the busy-clearing signal fire — the contract
    // that lets the composer recover from a hung/looping turn, not just the dot.
    expect($workingSessionIds.get()).not.toContain(sessionIdentityKey('s1', 'default'))
    expect(cleared).toEqual([sessionIdentityKey('s1', 'default')])

    off()
  })

  it('never fires for a session that settles before the window', () => {
    const cleared: string[] = []
    const off = onSessionWatchdogClear(id => cleared.push(id))

    setSessionWorking('s2', true)
    setSessionWorking('s2', false)

    vi.advanceTimersByTime(WATCHDOG_MS)

    expect(cleared).toEqual([])

    off()
  })

  it('stops notifying after unsubscribe', () => {
    const cleared: string[] = []
    const off = onSessionWatchdogClear(id => cleared.push(id))
    off()

    setSessionWorking('s3', true)
    vi.advanceTimersByTime(WATCHDOG_MS)

    expect(cleared).toEqual([])
  })
})
