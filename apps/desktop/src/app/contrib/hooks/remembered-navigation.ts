import {
  getRememberedRoute,
  getRememberedSessionId,
  sessionBelongsToProfile,
  sessionMatchesStoredId
} from '@/store/session'
import type { SessionInfo } from '@/types/hermes'

import { appViewForPath, isOverlayView, NEW_CHAT_ROUTE, routeSessionId, sessionRoute } from '../../routes'

export type RememberedSession = Pick<
  SessionInfo,
  '_lineage_root_id' | 'id' | 'parent_session_id' | 'profile' | 'source'
>

/** Pure validation shared with cold start; callers retain their own readiness
 * and side-effect ordering. By-id delegate repair uses remembered-session.ts. */
export function validateRememberedNavigation(
  route: string | null,
  last: string | null,
  profile: string,
  sessions: readonly RememberedSession[]
) {
  const routeSession = route ? routeSessionId(route) : null
  const rowFor = (id: string) => sessions.find(session => sessionMatchesStoredId(session, id))
  const belongs = (id: string) => sessionBelongsToProfile(sessions, id, profile)
  const listed = (id: string) => rowFor(id)?.source !== 'subagent' && belongs(id)
  const nonSessionRoute = !!route && route !== NEW_CHAT_ROUTE && !routeSession && !isOverlayView(appViewForPath(route))

  const validRoute =
    !!route &&
    route !== NEW_CHAT_ROUTE &&
    !isOverlayView(appViewForPath(route)) &&
    (!routeSession || listed(routeSession))

  return { routeSession, nonSessionRoute, validRoute, listedLast: !!last && listed(last), belongs, rowFor }
}

/** The hash commits synchronously before the router render. The history barrier
 * holds if either says new-chat (||); restore waits until both agree (&&). */
export function isMainOnNewChatRoute(pathname: string, phase: 'barrier' | 'restore'): boolean {
  const routerIsNew = pathname === NEW_CHAT_ROUTE
  const hashIsNew = window.location.hash.slice(1).split('?')[0] === NEW_CHAT_ROUTE

  return phase === 'barrier' ? routerIsNew || hashIsNew : routerIsNew && hashIsNew
}

interface RememberedNavigationOptions {
  profile: string
  sessions: readonly RememberedSession[]
  diskPluginsScanPending: boolean
  resumeExhaustedSessionId: null | string
  onDestination: (route: null | string, sessionId: null | string) => void
}

/** Live switch side effects, after scoped list authority. Cold start shares
 * the pure validator above but retains its original effect ordering. This
 * ladder never writes remembered navigation, even for unlisted/missing rows. */
export function restoreRememberedNavigation({
  profile,
  sessions,
  diskPluginsScanPending,
  resumeExhaustedSessionId,
  onDestination
}: RememberedNavigationOptions): void {
  const route = getRememberedRoute(profile)
  const last = getRememberedSessionId(profile)

  const { routeSession, validRoute, listedLast } = validateRememberedNavigation(route, last, profile, sessions)

  if (routeSession && diskPluginsScanPending) {
    return
  }

  if (validRoute && (!routeSession || routeSession !== resumeExhaustedSessionId)) {
    onDestination(route, routeSession)

    return
  }

  if (!last || last === resumeExhaustedSessionId) {
    onDestination(null, null)

    return
  }

  if (listedLast) {
    onDestination(sessionRoute(last), last)

    return
  }

  onDestination(null, null)
}
