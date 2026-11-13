import { cleanup, render } from '@testing-library/react'
import type { MutableRefObject } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $resumeExhaustedSessionId, setResumeExhaustedSessionId, setResumeReattachSessionId } from '@/store/session'

import { useRouteResume } from './use-route-resume'

vi.mock('@/store/session-states', () => ({ markSelectionRestore: vi.fn() }))

type Props = Parameters<typeof useRouteResume>[0]

function Harness(props: Props) {
  useRouteResume(props)

  return null
}

/**
 * A reload whose first session.resume fails while the REST fallback paints the
 * history leaves a readable chat bound to no runtime: the turn still running
 * there never streamed its reply in (stuck on "Result unavailable").
 */
describe('a failed resume that still painted the history', () => {
  afterEach(() => {
    cleanup()
    vi.useRealTimers()
    setResumeReattachSessionId(null)
    setResumeExhaustedSessionId(null)
  })

  it('skips a timer that fires after the selection moved or a runtime bound, and stays bounded', () => {
    vi.useFakeTimers()
    const resumeSession = vi.fn(async () => undefined)
    const activeSessionIdRef: MutableRefObject<null | string> = { current: null }
    const selectedStoredSessionIdRef: MutableRefObject<null | string> = { current: 'session-1' }

    const props: Props = {
      activeSessionId: null,
      activeSessionIdRef,
      creatingSessionRef: { current: false },
      currentView: 'chat',
      freshDraftReady: false,
      gatewayState: 'open',
      locationPathname: '/session-1',
      resumeSession,
      resumeFailedSessionId: null,
      resumeExhaustedSessionId: null,
      sessionResumeRequest: null,
      routedSessionId: 'session-1',
      runtimeIdByStoredSessionIdRef: { current: new Map<string, string>() },
      selectedStoredSessionId: 'session-1',
      selectedStoredSessionIdRef,
      startFreshSessionDraft: vi.fn()
    }

    const { rerender } = render(<Harness {...props} />)
    resumeSession.mockClear() // the mount resume is the one that failed

    // resumeSession arms the latch on that failure, and clears it at the
    // entry of every later attempt, which then fails the same way.
    const failAgain = () => {
      setResumeReattachSessionId(null)
      rerender(<Harness {...props} />)
      setResumeReattachSessionId('session-1')
      rerender(<Harness {...props} />)
    }

    // The refs move before the route/props catch up (renderer timers lag).
    failAgain()
    selectedStoredSessionIdRef.current = 'session-2'
    vi.advanceTimersByTime(1_000)
    selectedStoredSessionIdRef.current = 'session-1'

    failAgain()
    activeSessionIdRef.current = 'runtime-1'
    vi.advanceTimersByTime(1_000)
    activeSessionIdRef.current = null

    expect(resumeSession).not.toHaveBeenCalled()

    failAgain()
    vi.advanceTimersByTime(1_000)
    expect(resumeSession).toHaveBeenCalledWith('session-1', true, undefined, { reattach: true })

    for (let i = 0; i < 8; i += 1) {
      failAgain()
      vi.advanceTimersByTime(8_000)
    }

    expect(resumeSession).toHaveBeenCalledTimes(4)
    expect($resumeExhaustedSessionId.get()).toBeNull()
  })
})
