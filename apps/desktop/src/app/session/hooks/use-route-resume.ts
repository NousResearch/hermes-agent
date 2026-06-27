import { type MutableRefObject, useEffect, useRef } from 'react'

import { isNewChatRoute } from '@/app/routes'
import { setResumeExhaustedSessionId } from '@/store/session'

interface RouteResumeOptions {
  activeSessionId: string | null
  activeSessionIdRef: MutableRefObject<string | null>
  creatingSessionRef: MutableRefObject<boolean>
  currentView: string
  freshDraftReady: boolean
  gatewayState: string | undefined
  locationPathname: string
  resumeSession: (sessionId: string, focus: boolean) => Promise<unknown>
  // Stored-session id whose most recent resume failed terminally (set by
  // useSessionActions, mirrored from $resumeFailedSessionId). While this equals
  // routedSessionId the window would otherwise latch on the loader forever, so
  // the bounded-retry effect below re-attempts the resume.
  resumeFailedSessionId: string | null
  // Stored-session id whose bounded auto-retry has EXHAUSTED (mirrored from
  // $resumeExhaustedSessionId). Only resumeSession clears this latch (manual
  // Retry / reconnect / reselect) — the auto-retry loop never does — so its
  // armed->cleared edge is an unambiguous "give me a fresh backoff cycle"
  // signal the effect below uses to reset the attempt counter.
  resumeExhaustedSessionId: string | null
  routedSessionId: string | null
  runtimeIdByStoredSessionIdRef: MutableRefObject<Map<string, string>>
  selectedStoredSessionId: string | null
  selectedStoredSessionIdRef: MutableRefObject<string | null>
  startFreshSessionDraft: (focus: boolean) => unknown
}

// Bounded auto-retry for a stranded session window. A resume can fail terminally
// (gateway RPC reject + REST fallback failure) on a transiently wedged backend —
// dead provider key, a runaway turn hogging the dispatcher, flaky DNS. Without a
// retry the loader latches forever. We retry with backoff, capped, so a
// genuinely dead backend doesn't hot-loop the resume.
const MAX_RESUME_RETRIES = 4
const RESUME_RETRY_BASE_MS = 1_000
const RESUME_RETRY_MAX_MS = 8_000

function resumeRetryDelayMs(attempt: number): number {
  return Math.min(RESUME_RETRY_MAX_MS, RESUME_RETRY_BASE_MS * 2 ** attempt)
}

// HashRouter boot edge case: pathname briefly reads `/` before the hash is
// parsed. If the hash references a real session, defer; resume picks it up
// next tick. Without this, ctrl+R on `#/:sessionId` flashes 5 loading states.
function rawHashLooksLikeSession(): boolean {
  if (typeof window === 'undefined') {
    return false
  }

  const hash = window.location.hash.replace(/^#/, '')

  if (!hash || hash === '/') {
    return false
  }

  return (
    !hash.startsWith('/settings') &&
    !hash.startsWith('/skills') &&
    !hash.startsWith('/messaging') &&
    !hash.startsWith('/artifacts')
  )
}

export function useRouteResume({
  activeSessionId,
  activeSessionIdRef,
  creatingSessionRef,
  currentView,
  freshDraftReady,
  gatewayState,
  locationPathname,
  resumeSession,
  resumeFailedSessionId,
  resumeExhaustedSessionId,
  routedSessionId,
  runtimeIdByStoredSessionIdRef,
  selectedStoredSessionId,
  selectedStoredSessionIdRef,
  startFreshSessionDraft
}: RouteResumeOptions) {
  const lastPathnameRef = useRef<string | null>(null)
  const seenGatewayStateRef = useRef(false)
  const wasGatewayOpenRef = useRef(false)
  // Per-session retry bookkeeping for the bounded auto-retry effect below. Keyed
  // by the session id we're retrying so switching chats resets the counter.
  const retrySessionIdRef = useRef<string | null>(null)
  const retryAttemptRef = useRef(0)
  // Tracks the previous exhausted-latch value so we can detect its armed->cleared
  // edge. resumeSession clears $resumeExhaustedSessionId on a manual Retry /
  // reconnect / reselect; that transition is our cue to reset the attempt counter
  // for a fresh backoff cycle on the SAME session (the auto-retry loop itself
  // never touches this latch, so it can't spuriously trigger the reset).
  const prevResumeExhaustedRef = useRef<string | null>(null)

  useEffect(() => {
    const gatewayOpen = gatewayState === 'open'
    const pathnameChanged = lastPathnameRef.current !== locationPathname
    // Fire only on a genuine closed->open transition (a reconnect). seenGatewayStateRef
    // stays false until the first effect run, so a session that mounts with the gateway
    // already open is not mistaken for "became open" and does not double-resume with the
    // pathname-driven initial resume below.
    const gatewayBecameOpen = seenGatewayStateRef.current && !wasGatewayOpenRef.current && gatewayOpen
    lastPathnameRef.current = locationPathname
    seenGatewayStateRef.current = true
    wasGatewayOpenRef.current = gatewayOpen

    if (currentView !== 'chat' || !gatewayOpen) {
      return
    }

    if (routedSessionId) {
      const cachedRuntime = runtimeIdByStoredSessionIdRef.current.get(routedSessionId)

      const alreadyActive =
        routedSessionId === selectedStoredSessionIdRef.current &&
        Boolean(cachedRuntime) &&
        cachedRuntime === activeSessionIdRef.current

      // Self-heal a desynced view: the route points at a session that isn't the
      // loaded one. A create/stream race can leave selected/active null while
      // the route stays on /:sid (symptom: brand-new chat shows "Thinking" then
      // an empty transcript even though the turn completed and persisted). The
      // pathname didn't change, so the normal gate would skip and the view stays
      // stuck empty forever. selectedStoredSessionIdRef is set synchronously at
      // resume entry, so this can't loop; the resume's cached fast-path restores
      // the already-streamed messages without a refetch.
      //
      // Crucially this must NOT fire during a /:sid -> /new transition, where
      // startFreshSessionDraft nulls selected/active one render before the
      // pathname flips to / (same null+/:sid signature). freshDraftReady is the
      // discriminator: it's true while heading into a blank new chat, false when
      // genuinely stranded on a routed session.
      const stuckOnRoutedSession = routedSessionId !== selectedStoredSessionIdRef.current && !freshDraftReady

      // Resume when the route meaningfully changed, the gateway just opened, or
      // we're stranded on a routed session that never loaded. The first two
      // guard against a transient /:sid re-resume during "new chat" state clears
      // before the pathname updates from /:sid -> /.
      const shouldResume = pathnameChanged || gatewayBecameOpen || stuckOnRoutedSession

      // On a reconnect (gatewayBecameOpen) re-resume even when the route looks
      // `alreadyActive`: the cached runtime id can be stale once the gateway
      // rebinds/reaps the session on its side, and trusting it strands Desktop on
      // a dead id ("session not found"). Otherwise keep skipping when already active.
      if ((gatewayBecameOpen || !alreadyActive) && shouldResume && !creatingSessionRef.current) {
        void resumeSession(routedSessionId, true)
      }

      return
    }

    if (
      isNewChatRoute(locationPathname) &&
      !creatingSessionRef.current &&
      (selectedStoredSessionId || activeSessionId || !freshDraftReady) &&
      !rawHashLooksLikeSession()
    ) {
      startFreshSessionDraft(true)
    }
  }, [
    activeSessionId,
    activeSessionIdRef,
    creatingSessionRef,
    currentView,
    freshDraftReady,
    gatewayState,
    locationPathname,
    resumeSession,
    routedSessionId,
    runtimeIdByStoredSessionIdRef,
    selectedStoredSessionId,
    selectedStoredSessionIdRef,
    startFreshSessionDraft
  ])

  // Bounded auto-retry: when the routed session's resume failed terminally
  // (resumeFailedSessionId matches the route), schedule a backoff retry so the
  // window recovers on its own instead of latching the loader forever. This is
  // the safety net the main effect above can't provide: after a failed resume,
  // selectedStoredSessionIdRef.current already equals the route (resumeSession
  // sets it synchronously at entry) and the pathname/gateway are unchanged, so
  // none of stuckOnRoutedSession / pathnameChanged / gatewayBecameOpen fire
  // again. resumeSession clears resumeFailedSessionId on its next attempt; a
  // success keeps it clear (the effect's guard then no-ops), a repeat failure
  // re-arms it and we back off further, capped at MAX_RESUME_RETRIES.
  useEffect(() => {
    // Detect the exhausted-latch armed->cleared edge for the current route. Only
    // resumeSession clears $resumeExhaustedSessionId (manual Retry / reconnect /
    // reselect) — the auto-retry loop never touches it — so this transition
    // uniquely means "the user asked for another go." Reset the attempt counter
    // for a fresh bounded backoff cycle on the SAME session. Without this,
    // retryAttemptRef stays pinned at MAX after exhaustion (the !stranded reset
    // below only fires on a route CHANGE to a different session), so a manual
    // retry on the same stranded session would get exactly ONE attempt and then
    // immediately re-arm the exhausted error — never the renewed backoff cycle
    // the store/session.ts + use-session-actions.ts comments promise. (Point 2)
    const wasExhausted = prevResumeExhaustedRef.current
    prevResumeExhaustedRef.current = resumeExhaustedSessionId

    if (wasExhausted && wasExhausted === routedSessionId && resumeExhaustedSessionId !== wasExhausted) {
      retrySessionIdRef.current = routedSessionId
      retryAttemptRef.current = 0
    }

    if (currentView !== 'chat' || gatewayState !== 'open') {
      return
    }

    const stranded =
      Boolean(routedSessionId) && resumeFailedSessionId === routedSessionId && !creatingSessionRef.current

    if (!stranded) {
      // Route moved off the stranded session (or it recovered) — reset the
      // counter so a future failure on another session starts fresh, and clear
      // any exhausted-latch armed for a session we're no longer viewing (never
      // the current route: that's the error state we want to keep showing).
      // resumeSession also clears it on a fresh attempt; this covers a plain
      // route-change away from the stranded window.
      if (retrySessionIdRef.current !== routedSessionId) {
        retrySessionIdRef.current = null
        retryAttemptRef.current = 0
        setResumeExhaustedSessionId(current => (current && current !== routedSessionId ? null : current))
      }

      return
    }

    // New stranded session id → reset the attempt counter.
    if (retrySessionIdRef.current !== routedSessionId) {
      retrySessionIdRef.current = routedSessionId
      retryAttemptRef.current = 0
    }

    if (retryAttemptRef.current >= MAX_RESUME_RETRIES) {
      // Give up auto-retrying a persistently dead backend; the user can still
      // reconnect / reselect (which resets the counter via the branch above).
      // Surface an explicit error + manual Retry in the chat view instead of
      // spinning the loader forever — resumeSession (manual Retry / reconnect /
      // reselect) clears this latch and resets the counter for a fresh cycle.
      setResumeExhaustedSessionId(routedSessionId)

      return
    }

    const attempt = retryAttemptRef.current
    const sessionId = routedSessionId as string

    const timer = setTimeout(() => {
      // Re-check liveness at fire time: a resume may have landed while we waited.
      if (
        creatingSessionRef.current ||
        selectedStoredSessionIdRef.current !== sessionId ||
        activeSessionIdRef.current !== null
      ) {
        return
      }

      // Consume an attempt ONLY now that a resume is actually dispatching.
      // Incrementing at schedule time (the old behavior) let unrelated dep
      // changes during the 1s–8s backoff window — a transient gatewayState
      // flip, a non-referentially-stable resumeSession — clear the pending
      // timer and re-run the effect, burning an attempt without any resume
      // having fired. A flapping backend could then hit MAX in a couple of
      // re-renders with far fewer than MAX real attempts. (Point 3)
      retryAttemptRef.current += 1
      void resumeSession(sessionId, true)
    }, resumeRetryDelayMs(attempt))

    return () => clearTimeout(timer)
  }, [
    activeSessionIdRef,
    creatingSessionRef,
    currentView,
    gatewayState,
    resumeSession,
    resumeFailedSessionId,
    resumeExhaustedSessionId,
    routedSessionId,
    selectedStoredSessionIdRef
  ])
}
