import { afterEach, describe, expect, it, vi } from 'vitest'

import { $workingSessionIds, setSessionWorking } from './session'

describe('session working watchdog', () => {
  afterEach(() => {
    setSessionWorking('background-profile-session', false)
    vi.useRealTimers()
    $workingSessionIds.set([])
  })

  it('keeps a silent running turn alive through the desktop backend idle window', () => {
    vi.useFakeTimers()
    vi.setSystemTime(0)
    $workingSessionIds.set([])

    setSessionWorking('background-profile-session', true)

    // Electron's pooled profile backend idle reaper fires at 10 minutes. A
    // legitimate long-running command can be silent for that whole window, so
    // the renderer watchdog must not drop the working flag before the backend
    // gets another keepalive tick.
    vi.advanceTimersByTime(10 * 60 * 1000)

    expect($workingSessionIds.get()).toContain('background-profile-session')
  })
})
