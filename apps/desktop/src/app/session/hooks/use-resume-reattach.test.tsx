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

  it('re-attaches on backoff, bounded, and never covers the history with the exhausted error', () => {
    vi.useFakeTimers()
    const resumeSession = vi.fn(async () => undefined)

    const props: Props = {
      activeSessionId: null,
      activeSessionIdRef: { current: null } as MutableRefObject<null | string>,
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
      selectedStoredSessionIdRef: { current: 'session-1' } as MutableRefObject<null | string>,
      startFreshSessionDraft: vi.fn()
    }

    const { rerender } = render(<Harness {...props} />)
    resumeSession.mockClear() // the mount resume is the one that failed

    // What resumeSession does on that failure: history painted, latch armed.
    setResumeReattachSessionId('session-1')
    rerender(<Harness {...props} />)

    vi.advanceTimersByTime(1_000)
    expect(resumeSession).toHaveBeenCalledTimes(1)
    expect(resumeSession).toHaveBeenCalledWith('session-1', true)

    // Every retry clears the latch at entry and fails the same way again.
    for (let i = 0; i < 8; i += 1) {
      setResumeReattachSessionId(null)
      rerender(<Harness {...props} />)
      setResumeReattachSessionId('session-1')
      rerender(<Harness {...props} />)
      vi.advanceTimersByTime(8_000)
    }

    expect(resumeSession).toHaveBeenCalledTimes(4)
    expect($resumeExhaustedSessionId.get()).toBeNull()
  })
})
