import { cleanup, render } from '@testing-library/react'
import type { MutableRefObject } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { useRouteResume } from './use-route-resume'

interface HarnessProps {
  activeSessionId: null | string
  activeSessionIdRef: MutableRefObject<null | string>
  creatingSessionRef: MutableRefObject<boolean>
  currentView: string
  freshDraftReady: boolean
  gatewayState: string
  locationPathname: string
  resumeSession: (sessionId: string, focus: boolean) => Promise<unknown>
  routedSessionId: null | string
  runtimeIdByStoredSessionIdRef: MutableRefObject<Map<string, string>>
  selectedStoredSessionId: null | string
  selectedStoredSessionIdRef: MutableRefObject<null | string>
  startFreshSessionDraft: (focus: boolean) => unknown
}

function RouteResumeHarness(props: HarnessProps) {
  useRouteResume(props)

  return null
}

describe('useRouteResume', () => {
  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('does not re-resume the old session during a /:sid -> /new transition', () => {
    const resumeSession = vi.fn(async () => undefined)
    const startFreshSessionDraft = vi.fn()
    const activeSessionIdRef: MutableRefObject<null | string> = { current: 'runtime-1' }
    const creatingSessionRef = { current: false }
    const runtimeIdByStoredSessionIdRef = { current: new Map([['session-1', 'runtime-1']]) }
    const selectedStoredSessionIdRef: MutableRefObject<null | string> = { current: 'session-1' }

    const { rerender } = render(
      <RouteResumeHarness
        activeSessionId="runtime-1"
        activeSessionIdRef={activeSessionIdRef}
        creatingSessionRef={creatingSessionRef}
        currentView="chat"
        freshDraftReady={false}
        gatewayState="open"
        locationPathname="/session-1"
        resumeSession={resumeSession}
        routedSessionId="session-1"
        runtimeIdByStoredSessionIdRef={runtimeIdByStoredSessionIdRef}
        selectedStoredSessionId="session-1"
        selectedStoredSessionIdRef={selectedStoredSessionIdRef}
        startFreshSessionDraft={startFreshSessionDraft}
      />
    )

    expect(resumeSession).not.toHaveBeenCalled()

    // Simulate startFreshSessionDraft state updates landing before route update.
    activeSessionIdRef.current = null
    selectedStoredSessionIdRef.current = null
    rerender(
      <RouteResumeHarness
        activeSessionId={null}
        activeSessionIdRef={activeSessionIdRef}
        creatingSessionRef={creatingSessionRef}
        currentView="chat"
        freshDraftReady
        gatewayState="open"
        locationPathname="/session-1"
        resumeSession={resumeSession}
        routedSessionId="session-1"
        runtimeIdByStoredSessionIdRef={runtimeIdByStoredSessionIdRef}
        selectedStoredSessionId={null}
        selectedStoredSessionIdRef={selectedStoredSessionIdRef}
        startFreshSessionDraft={startFreshSessionDraft}
      />
    )

    expect(resumeSession).not.toHaveBeenCalled()
  })

  it('resumes when pathname changes to a routed session', () => {
    const resumeSession = vi.fn(async () => undefined)
    const startFreshSessionDraft = vi.fn()
    const activeSessionIdRef: MutableRefObject<null | string> = { current: null }
    const creatingSessionRef = { current: false }
    const runtimeIdByStoredSessionIdRef = { current: new Map() }
    const selectedStoredSessionIdRef: MutableRefObject<null | string> = { current: null }

    const { rerender } = render(
      <RouteResumeHarness
        activeSessionId={null}
        activeSessionIdRef={activeSessionIdRef}
        creatingSessionRef={creatingSessionRef}
        currentView="chat"
        freshDraftReady
        gatewayState="open"
        locationPathname="/"
        resumeSession={resumeSession}
        routedSessionId={null}
        runtimeIdByStoredSessionIdRef={runtimeIdByStoredSessionIdRef}
        selectedStoredSessionId={null}
        selectedStoredSessionIdRef={selectedStoredSessionIdRef}
        startFreshSessionDraft={startFreshSessionDraft}
      />
    )

    expect(resumeSession).not.toHaveBeenCalled()

    rerender(
      <RouteResumeHarness
        activeSessionId={null}
        activeSessionIdRef={activeSessionIdRef}
        creatingSessionRef={creatingSessionRef}
        currentView="chat"
        freshDraftReady
        gatewayState="open"
        locationPathname="/session-2"
        resumeSession={resumeSession}
        routedSessionId="session-2"
        runtimeIdByStoredSessionIdRef={runtimeIdByStoredSessionIdRef}
        selectedStoredSessionId={null}
        selectedStoredSessionIdRef={selectedStoredSessionIdRef}
        startFreshSessionDraft={startFreshSessionDraft}
      />
    )

    expect(resumeSession).toHaveBeenCalledTimes(1)
    expect(resumeSession).toHaveBeenCalledWith('session-2', true)
  })
})
