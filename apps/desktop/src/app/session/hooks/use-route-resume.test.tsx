import { cleanup, render } from '@testing-library/react'
import type { MutableRefObject } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $resumeExhaustedSessionId, setResumeExhaustedSessionId } from '@/store/session'

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
  resumeFailedSessionId?: null | string
  resumeExhaustedSessionId?: null | string
  routedSessionId: null | string
  routingReady?: boolean
  runtimeIdByStoredSessionIdRef: MutableRefObject<Map<string, string>>
  selectedStoredSessionId: null | string
  selectedStoredSessionIdRef: MutableRefObject<null | string>
  startFreshSessionDraft: (focus: boolean) => unknown
}

function RouteResumeHarness({
  resumeFailedSessionId = null,
  resumeExhaustedSessionId = null,
  routingReady = true,
  ...props
}: HarnessProps) {
  useRouteResume({ ...props, resumeExhaustedSessionId, resumeFailedSessionId, routingReady })

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

  it('waits for profile routing readiness before resuming a hard-refreshed route', () => {
    const resumeSession = vi.fn(async () => undefined)
    const startFreshSessionDraft = vi.fn()
    const activeSessionIdRef: MutableRefObject<null | string> = { current: null }
    const creatingSessionRef = { current: false }
    const runtimeIdByStoredSessionIdRef = { current: new Map<string, string>() }
    const selectedStoredSessionIdRef: MutableRefObject<null | string> = { current: null }

    const props = {
      activeSessionId: null,
      activeSessionIdRef,
      creatingSessionRef,
      currentView: 'chat',
      freshDraftReady: false,
      gatewayState: 'open',
      locationPathname: '/session-profiled',
      resumeSession,
      routedSessionId: 'session-profiled',
      runtimeIdByStoredSessionIdRef,
      selectedStoredSessionId: null,
      selectedStoredSessionIdRef,
      startFreshSessionDraft
    } satisfies Omit<HarnessProps, 'routingReady'>

    const { rerender } = render(<RouteResumeHarness {...props} routingReady={false} />)

    expect(resumeSession).not.toHaveBeenCalled()

    rerender(<RouteResumeHarness {...props} routingReady />)

    expect(resumeSession).toHaveBeenCalledTimes(1)
    expect(resumeSession).toHaveBeenCalledWith('session-profiled', true)
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

describe('useRouteResume bounded auto-retry after a failed resume', () => {
  afterEach(() => {
    cleanup()
    vi.useRealTimers()
    vi.restoreAllMocks()
    setResumeExhaustedSessionId(null)
  })

  // Common stranded-window props: gateway open, route on the session, no runtime
  // yet, and the ref already synced to the route (resumeSession sets it at entry
  // before failing) — the exact state that defeats the main effect's self-heal.
  function strandedProps(resumeSession: (sid: string, focus: boolean) => Promise<unknown>) {
    return {
      activeSessionId: null,
      activeSessionIdRef: { current: null } as MutableRefObject<null | string>,
      creatingSessionRef: { current: false },
      currentView: 'chat',
      freshDraftReady: false,
      gatewayState: 'open',
      locationPathname: '/session-1',
      resumeSession,
      routedSessionId: 'session-1',
      runtimeIdByStoredSessionIdRef: { current: new Map<string, string>() },
      selectedStoredSessionId: 'session-1',
      // Synced to the route by the failed resume's synchronous entry-write.
      selectedStoredSessionIdRef: { current: 'session-1' } as MutableRefObject<null | string>,
      startFreshSessionDraft: vi.fn()
    }
  }

  it('retries the resume on backoff when the routed session is flagged as failed', () => {
    vi.useFakeTimers()
    const resumeSession = vi.fn(async () => undefined)

    render(<RouteResumeHarness {...strandedProps(resumeSession)} resumeFailedSessionId="session-1" />)

    // The main effect fires one resume on mount (pathname-changed). Clear it so
    // we assert purely the bounded-retry effect's scheduled retry below.
    resumeSession.mockClear()

    // No immediate fire — the retry is scheduled behind the backoff timer.
    expect(resumeSession).not.toHaveBeenCalled()

    // First backoff window (1s) elapses → one retry.
    vi.advanceTimersByTime(1_000)
    expect(resumeSession).toHaveBeenCalledTimes(1)
    expect(resumeSession).toHaveBeenCalledWith('session-1', true)
  })

  it('does NOT retry a failed session that is not the routed one', () => {
    vi.useFakeTimers()
    const resumeSession = vi.fn(async () => undefined)

    // The failure flag points at a different session than the route.
    render(<RouteResumeHarness {...strandedProps(resumeSession)} resumeFailedSessionId="other-session" />)
    resumeSession.mockClear() // drop the mount resume

    vi.advanceTimersByTime(10_000)
    expect(resumeSession).not.toHaveBeenCalled()
  })

  it('skips the scheduled retry if the session already recovered when the timer fires', () => {
    vi.useFakeTimers()
    const resumeSession = vi.fn(async () => undefined)
    const props = strandedProps(resumeSession)

    render(<RouteResumeHarness {...props} resumeFailedSessionId="session-1" />)
    resumeSession.mockClear() // drop the mount resume

    // A resume landed while we waited: runtime is now bound.
    props.activeSessionIdRef.current = 'runtime-1'

    vi.advanceTimersByTime(8_000)
    expect(resumeSession).not.toHaveBeenCalled()
  })

  it('stops retrying after MAX_RESUME_RETRIES consecutive failures', () => {
    vi.useFakeTimers()
    const resumeSession = vi.fn(async () => undefined)
    const props = strandedProps(resumeSession)

    // Model the real re-arm loop: resumeSession clears $resumeFailedSessionId at
    // entry (null) and a repeat failure re-sets it ('session-1'). That null->id
    // toggle is what re-runs the effect and advances the bounded counter. The
    // routed session never changes, so the counter is NOT reset between cycles.
    const { rerender } = render(<RouteResumeHarness {...props} resumeFailedSessionId="session-1" />)
    resumeSession.mockClear() // drop the mount resume; count only the retries

    for (let i = 0; i < 8; i += 1) {
      vi.advanceTimersByTime(8_000) // fire the scheduled retry (if any)
      rerender(<RouteResumeHarness {...props} resumeFailedSessionId={null} />) // cleared at entry
      rerender(<RouteResumeHarness {...props} resumeFailedSessionId="session-1" />) // re-armed on failure
    }

    // Capped at MAX_RESUME_RETRIES (4): a persistently dead backend can't
    // hot-loop the resume forever.
    expect(resumeSession.mock.calls.length).toBe(4)

    // Once auto-retry gives up, the exhausted latch is armed for the routed
    // session so the chat view can swap the perpetual loader for an explicit
    // error + manual Retry instead of spinning forever.
    expect($resumeExhaustedSessionId.get()).toBe('session-1')
  })

  it('does not arm the exhausted latch while retries remain', () => {
    vi.useFakeTimers()
    const resumeSession = vi.fn(async () => undefined)
    const props = strandedProps(resumeSession)

    const { rerender } = render(<RouteResumeHarness {...props} resumeFailedSessionId="session-1" />)
    resumeSession.mockClear()

    // Two failure cycles — still under the 4-retry cap, so the latch must stay
    // clear and the loader keeps spinning (auto-recovery hasn't given up yet).
    for (let i = 0; i < 2; i += 1) {
      vi.advanceTimersByTime(8_000)
      rerender(<RouteResumeHarness {...props} resumeFailedSessionId={null} />)
      rerender(<RouteResumeHarness {...props} resumeFailedSessionId="session-1" />)
    }

    expect($resumeExhaustedSessionId.get()).toBeNull()
  })

  it('clears a stale exhausted latch when the route moves off the stranded session', () => {
    vi.useFakeTimers()
    const resumeSession = vi.fn(async () => undefined)
    const props = strandedProps(resumeSession)

    // Pre-arm the latch as if this session had exhausted its retries.
    setResumeExhaustedSessionId('session-1')

    // Route is now on a different, healthy session that is not flagged as
    // failed — the retry effect's "route moved off" branch clears the latch.
    render(
      <RouteResumeHarness
        {...props}
        activeSessionId="runtime-2"
        activeSessionIdRef={{ current: 'runtime-2' }}
        locationPathname="/session-2"
        resumeFailedSessionId={null}
        routedSessionId="session-2"
        selectedStoredSessionId="session-2"
        selectedStoredSessionIdRef={{ current: 'session-2' }}
      />
    )

    expect($resumeExhaustedSessionId.get()).toBeNull()
  })

  it('resets the retry counter for a fresh backoff cycle when the exhausted latch clears (manual retry, same session)', () => {
    vi.useFakeTimers()
    const resumeSession = vi.fn(async () => undefined)
    const props = strandedProps(resumeSession)

    // Phase A — exhaust the bounded auto-retry (counter → MAX) like a dead
    // backend. The resumeExhaustedSessionId prop stays null here: the hook sets
    // the store, which doesn't feed back into the prop in this harness.
    const { rerender } = render(<RouteResumeHarness {...props} resumeFailedSessionId="session-1" />)
    resumeSession.mockClear()

    for (let i = 0; i < 8; i += 1) {
      vi.advanceTimersByTime(8_000)
      rerender(<RouteResumeHarness {...props} resumeFailedSessionId={null} />)
      rerender(<RouteResumeHarness {...props} resumeFailedSessionId="session-1" />)
    }

    expect(resumeSession.mock.calls.length).toBe(4) // capped
    expect($resumeExhaustedSessionId.get()).toBe('session-1')

    // Phase B — user clicks Retry on the SAME stranded session. resumeSession
    // clears both latches at entry; the exhausted latch's armed->cleared edge
    // must reset the attempt counter so a fresh bounded cycle runs, not a single
    // one-shot attempt that immediately re-arms the error. Model the prop
    // transitions: reflect the armed latch, then clear it (retry), then re-arm
    // the failure latch on the fresh failure.
    resumeSession.mockClear()
    rerender(<RouteResumeHarness {...props} resumeExhaustedSessionId="session-1" resumeFailedSessionId="session-1" />)
    rerender(<RouteResumeHarness {...props} resumeExhaustedSessionId={null} resumeFailedSessionId={null} />)
    rerender(<RouteResumeHarness {...props} resumeExhaustedSessionId={null} resumeFailedSessionId="session-1" />)

    // A real retry fires again instead of staying pinned at MAX (which would
    // dispatch nothing). Without the reset the counter stays >= MAX and this
    // advance dispatches zero resumes.
    vi.advanceTimersByTime(8_000)
    expect(resumeSession.mock.calls.length).toBeGreaterThan(0)
  })

  it('does not burn retry attempts on unrelated re-renders during the backoff window', () => {
    vi.useFakeTimers()
    const props = strandedProps(vi.fn())

    // Mount schedules the first backoff timer. Then re-render repeatedly with a
    // fresh resumeSession identity (referential instability — a real dep change
    // for the retry effect) WITHOUT ever letting the timer fire. The old code
    // incremented the attempt counter at schedule time, so >= MAX re-renders
    // armed the exhausted error with zero resumes actually dispatched. The fix
    // only advances the counter when a timer truly fires, so the latch stays
    // clear no matter how many spurious re-renders happen mid-backoff.
    const { rerender } = render(
      <RouteResumeHarness {...props} resumeFailedSessionId="session-1" resumeSession={vi.fn(async () => undefined)} />
    )

    for (let j = 0; j < 8; j += 1) {
      rerender(
        <RouteResumeHarness {...props} resumeFailedSessionId="session-1" resumeSession={vi.fn(async () => undefined)} />
      )
    }

    expect($resumeExhaustedSessionId.get()).toBeNull()
  })
})
