import { useStore } from '@nanostores/react'
import { type MutableRefObject, useEffect, useRef } from 'react'

import { dismissNotification } from '@/store/notifications'
import { $resumeReattachSessionId } from '@/store/session'

import {
  MAX_RESUME_RETRIES,
  type ResumeAttemptOptions,
  resumeFailedToastId,
  resumeRetryDelayMs
} from './use-session-actions/resume-latches'

interface ResumeReattachOptions {
  activeSessionId: string | null
  activeSessionIdRef: MutableRefObject<string | null>
  creatingSessionRef: MutableRefObject<boolean>
  currentView: string
  gatewayState: string | undefined
  resumeSession: (
    sessionId: string,
    focus: boolean,
    ownerRoute?: undefined,
    options?: ResumeAttemptOptions
  ) => Promise<unknown>
  routedSessionId: string | null
  selectedStoredSessionIdRef: MutableRefObject<string | null>
}

/**
 * Re-attach a routed chat whose resume RPC failed while the REST fallback
 * still painted its history ($resumeReattachSessionId). The window reads fine
 * but is bound to no runtime, and nothing else resumes it: the route, the
 * selection and the gateway are all unchanged. A turn still running there then
 * never streams its reply in, and the view keeps "Result unavailable" for good.
 * Retries back off and stop after MAX_RESUME_RETRIES; the history stays on
 * screen either way, and a reconnect or reselect resumes it as before. They
 * are quiet: no app-wide toast wipe, no repeat "Resume failed" (resume-latches).
 */
export function useResumeReattach({
  activeSessionId,
  activeSessionIdRef,
  creatingSessionRef,
  currentView,
  gatewayState,
  resumeSession,
  routedSessionId,
  selectedStoredSessionIdRef
}: ResumeReattachOptions): void {
  const reattachSessionId = useStore($resumeReattachSessionId)
  const attemptsRef = useRef<{ count: number; sessionId: null | string }>({ count: 0, sessionId: null })

  useEffect(() => {
    const attempts = attemptsRef.current

    // A retry re-attached: the incident's "Resume failed" toast is stale now
    // (retries no longer clear the app's toasts on entry).
    if (activeSessionId && routedSessionId && attempts.count > 0 && attempts.sessionId === routedSessionId) {
      dismissNotification(resumeFailedToastId(routedSessionId))
    }

    // A bound runtime (or another chat) starts the next detach on a fresh cycle.
    if (activeSessionId || attempts.sessionId !== routedSessionId) {
      attempts.sessionId = routedSessionId
      attempts.count = 0
    }

    if (
      currentView !== 'chat' ||
      gatewayState !== 'open' ||
      !routedSessionId ||
      reattachSessionId !== routedSessionId ||
      attempts.count >= MAX_RESUME_RETRIES
    ) {
      return
    }

    const sessionId = routedSessionId

    const timer = setTimeout(() => {
      // The resume may have landed, or the user moved on, while we waited.
      if (
        creatingSessionRef.current ||
        selectedStoredSessionIdRef.current !== sessionId ||
        activeSessionIdRef.current !== null
      ) {
        return
      }

      attempts.count += 1
      void resumeSession(sessionId, true, undefined, { reattach: true })
    }, resumeRetryDelayMs(attempts.count))

    return () => clearTimeout(timer)
  }, [
    activeSessionId,
    activeSessionIdRef,
    creatingSessionRef,
    currentView,
    gatewayState,
    reattachSessionId,
    resumeSession,
    routedSessionId,
    selectedStoredSessionIdRef
  ])
}
