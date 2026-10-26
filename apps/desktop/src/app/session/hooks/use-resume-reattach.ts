import { useStore } from '@nanostores/react'
import { type MutableRefObject, useEffect, useRef } from 'react'

import { $resumeReattachSessionId } from '@/store/session'

interface ResumeReattachOptions {
  activeSessionId: string | null
  activeSessionIdRef: MutableRefObject<string | null>
  creatingSessionRef: MutableRefObject<boolean>
  currentView: string
  gatewayState: string | undefined
  resumeSession: (sessionId: string, focus: boolean) => Promise<unknown>
  routedSessionId: string | null
  selectedStoredSessionIdRef: MutableRefObject<string | null>
}

const MAX_REATTACH_ATTEMPTS = 4
const REATTACH_BASE_MS = 1_000
const REATTACH_MAX_MS = 8_000

/**
 * Re-attach a routed chat whose resume RPC failed while the REST fallback
 * still painted its history ($resumeReattachSessionId). The window reads fine
 * but is bound to no runtime, and nothing else resumes it: the route, the
 * selection and the gateway are all unchanged. A turn still running there then
 * never streams its reply in, and the view keeps "Result unavailable" for good.
 * Retries back off and stop after MAX_REATTACH_ATTEMPTS; the history stays on
 * screen either way, and a reconnect or reselect resumes it as before.
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
      attempts.count >= MAX_REATTACH_ATTEMPTS
    ) {
      return
    }

    const sessionId = routedSessionId

    const timer = setTimeout(
      () => {
        // The resume may have landed, or the user moved on, while we waited.
        if (
          creatingSessionRef.current ||
          selectedStoredSessionIdRef.current !== sessionId ||
          activeSessionIdRef.current !== null
        ) {
          return
        }

        attempts.count += 1
        void resumeSession(sessionId, true)
      },
      Math.min(REATTACH_MAX_MS, REATTACH_BASE_MS * 2 ** attempts.count)
    )

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
