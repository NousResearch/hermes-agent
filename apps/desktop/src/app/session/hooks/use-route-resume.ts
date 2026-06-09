import { type MutableRefObject, useEffect, useRef } from 'react'

import { isNewChatRoute } from '@/app/routes'

interface RouteResumeOptions {
  activeSessionId: string | null
  activeSessionIdRef: MutableRefObject<string | null>
  creatingSessionRef: MutableRefObject<boolean>
  currentView: string
  freshDraftReady: boolean
  gatewayState: string | undefined
  locationPathname: string
  resumeSession: (sessionId: string, focus: boolean) => Promise<unknown>
  routedSessionId: string | null
  runtimeIdByStoredSessionIdRef: MutableRefObject<Map<string, string>>
  selectedStoredSessionId: string | null
  selectedStoredSessionIdRef: MutableRefObject<string | null>
  startFreshSessionDraft: (focus: boolean) => unknown
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
  routedSessionId,
  runtimeIdByStoredSessionIdRef,
  selectedStoredSessionId,
  selectedStoredSessionIdRef,
  startFreshSessionDraft
}: RouteResumeOptions) {
  const lastPathnameRef = useRef<string | null>(null)
  const wasGatewayOpenRef = useRef(false)

  useEffect(() => {
    const gatewayOpen = gatewayState === 'open'
    const pathnameChanged = lastPathnameRef.current !== locationPathname
    const gatewayBecameOpen = !wasGatewayOpenRef.current && gatewayOpen
    lastPathnameRef.current = locationPathname
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

      if (!alreadyActive && shouldResume && !creatingSessionRef.current) {
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
}
