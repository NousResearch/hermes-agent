import { cleanup, render } from '@testing-library/react'
import type { MutableRefObject } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { HermesConnection } from '@/global'
import { setConnection, setSelectedStoredSessionId } from '@/store/session'

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
    setConnection(null)
    setSelectedStoredSessionId(null)
    window.localStorage.clear()
    vi.restoreAllMocks()
  })

  it('resumes the remembered session on root startup', () => {
    const remoteConnection = {
      baseUrl: 'https://gateway.example',
      isFullscreen: false,
      logs: [],
      mode: 'remote',
      nativeOverlayWidth: 0,
      profile: 'default',
      token: '',
      windowButtonPosition: null,
      wsUrl: 'wss://gateway.example/ws'
    } satisfies HermesConnection

    setConnection(remoteConnection)
    setSelectedStoredSessionId('session-persisted')

    const resumeSession = vi.fn(async () => undefined)
    const startFreshSessionDraft = vi.fn()
    const activeSessionIdRef: MutableRefObject<null | string> = { current: null }
    const creatingSessionRef = { current: false }
    const runtimeIdByStoredSessionIdRef = { current: new Map() }
    const selectedStoredSessionIdRef: MutableRefObject<null | string> = { current: null }

    render(
      <RouteResumeHarness
        activeSessionId={null}
        activeSessionIdRef={activeSessionIdRef}
        creatingSessionRef={creatingSessionRef}
        currentView="chat"
        freshDraftReady={false}
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

    expect(resumeSession).toHaveBeenCalledTimes(1)
    expect(resumeSession).toHaveBeenCalledWith('session-persisted', true)
    expect(startFreshSessionDraft).not.toHaveBeenCalled()
  })

  it('starts a fresh draft on root startup when no session is remembered', () => {
    const resumeSession = vi.fn(async () => undefined)
    const startFreshSessionDraft = vi.fn()
    const activeSessionIdRef: MutableRefObject<null | string> = { current: null }
    const creatingSessionRef = { current: false }
    const runtimeIdByStoredSessionIdRef = { current: new Map() }
    const selectedStoredSessionIdRef: MutableRefObject<null | string> = { current: null }

    render(
      <RouteResumeHarness
        activeSessionId={null}
        activeSessionIdRef={activeSessionIdRef}
        creatingSessionRef={creatingSessionRef}
        currentView="chat"
        freshDraftReady={false}
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
    expect(startFreshSessionDraft).toHaveBeenCalledTimes(1)
    expect(startFreshSessionDraft).toHaveBeenCalledWith(true)
  })

  it('does not restore a remembered session while an explicit fresh draft is ready', () => {
    setSelectedStoredSessionId('session-persisted')

    const resumeSession = vi.fn(async () => undefined)
    const startFreshSessionDraft = vi.fn()
    const activeSessionIdRef: MutableRefObject<null | string> = { current: null }
    const creatingSessionRef = { current: false }
    const runtimeIdByStoredSessionIdRef = { current: new Map() }
    const selectedStoredSessionIdRef: MutableRefObject<null | string> = { current: null }

    render(
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
    expect(startFreshSessionDraft).not.toHaveBeenCalled()
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

  it('self-heals a stranded routed session (null selected/active, same pathname, not a fresh draft)', () => {
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

    // A create/stream race nulls selected/active but the route stays on the
    // session and freshDraftReady is false (NOT a new-chat transition).
    activeSessionIdRef.current = null
    selectedStoredSessionIdRef.current = null
    rerender(
      <RouteResumeHarness
        activeSessionId={null}
        activeSessionIdRef={activeSessionIdRef}
        creatingSessionRef={creatingSessionRef}
        currentView="chat"
        freshDraftReady={false}
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

    expect(resumeSession).toHaveBeenCalledTimes(1)
    expect(resumeSession).toHaveBeenCalledWith('session-1', true)
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

  it('resumes the selected route again when the gateway reconnects', () => {
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

    rerender(
      <RouteResumeHarness
        activeSessionId="runtime-1"
        activeSessionIdRef={activeSessionIdRef}
        creatingSessionRef={creatingSessionRef}
        currentView="chat"
        freshDraftReady={false}
        gatewayState="closed"
        locationPathname="/session-1"
        resumeSession={resumeSession}
        routedSessionId="session-1"
        runtimeIdByStoredSessionIdRef={runtimeIdByStoredSessionIdRef}
        selectedStoredSessionId="session-1"
        selectedStoredSessionIdRef={selectedStoredSessionIdRef}
        startFreshSessionDraft={startFreshSessionDraft}
      />
    )

    rerender(
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

    expect(resumeSession).toHaveBeenCalledTimes(1)
    expect(resumeSession).toHaveBeenCalledWith('session-1', true)
  })
})
