import { sessionBelongsToProfile, sessionMatchesStoredId } from '@/store/session'
import type { SessionInfo } from '@/types/hermes'

import { appViewForPath, isOverlayView, NEW_CHAT_ROUTE, routeSessionId } from '../../routes'

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
