import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $workingSessions, onSessionWatchdogClear, setSessionWorking, setWorkingSessions } from './session'
import type { SessionIdentity } from './session-identity'

const WATCHDOG_MS = 8 * 60 * 1000

describe('session watchdog', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    setWorkingSessions(() => [])
  })

  afterEach(() => {
    vi.runOnlyPendingTimers()
    vi.useRealTimers()
  })

  it('drops and notifies only the timed-out profile when stored ids collide', () => {
    const cleared: SessionIdentity[] = []
    const off = onSessionWatchdogClear(id => cleared.push(id))

    setSessionWorking('default', 'same-id', true)
    vi.advanceTimersByTime(1)
    setSessionWorking('work', 'same-id', true)

    vi.advanceTimersByTime(WATCHDOG_MS - 1)

    expect($workingSessions.get()).toEqual([{ profile: 'work', sessionId: 'same-id' }])
    expect(cleared).toEqual([{ profile: 'default', sessionId: 'same-id' }])

    off()
  })

  it('never fires for a session that settles before the window', () => {
    const cleared: SessionIdentity[] = []
    const off = onSessionWatchdogClear(id => cleared.push(id))

    setSessionWorking('default', 's2', true)
    setSessionWorking('default', 's2', false)

    vi.advanceTimersByTime(WATCHDOG_MS)

    expect(cleared).toEqual([])

    off()
  })

  it('stops notifying after unsubscribe', () => {
    const cleared: SessionIdentity[] = []
    const off = onSessionWatchdogClear(id => cleared.push(id))
    off()

    setSessionWorking('default', 's3', true)
    vi.advanceTimersByTime(WATCHDOG_MS)

    expect(cleared).toEqual([])
  })
})
