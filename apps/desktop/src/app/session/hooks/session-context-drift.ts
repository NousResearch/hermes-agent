import { isNewChatRoute, routeSessionId } from '../../routes'

/**
 * The chat a route token points at: the stored/routed session id, `'__new__'`
 * for the new-chat route, or null for a route that isn't a chat (settings and
 * the other overlay routes). Used to compare two route tokens by their *chat*
 * rather than their raw string.
 */
export type RouteTarget = string | null

/**
 * Reduce a route token to the chat it targets. The token is
 * `${pathname}:${search}:${hash}` (desktop-controller's routeToken), and only
 * the pathname selects the chat — search/hash carry overlay/panel state, so a
 * change there must not read as a session switch. We take the substring before
 * the first ':' as the pathname; that is safe because `location.pathname` never
 * contains a raw ':' (sessionRoute encodeURIComponent's the id, so a ':' in an
 * id arrives as %3A, and the app's other routes are literal colon-free paths).
 */
export function routeTargetFromToken(token: string): RouteTarget {
  const separator = token.indexOf(':')
  const pathname = separator === -1 ? token : token.slice(0, separator)

  return routeSessionId(pathname) ?? (isNewChatRoute(pathname) ? '__new__' : null)
}

interface SessionContextDriftArgs {
  startRouteToken: string
  nowRouteToken: string
  startSelectedStoredId: string | null
  nowSelectedStoredId: string | null
  /**
   * The stored session this submit is bound to, when known. Drift ignores a
   * move *to* this id: the submit pipeline itself re-homes selection and route
   * onto its target (a fresh create, a resume), and that self-inflicted move is
   * not a user switch. Omit it (pre-create new-chat draft) to treat any move to
   * a real chat as drift.
   */
  submitTargetStoredId?: string | null
}

/**
 * Decide whether the session context genuinely changed under an in-flight
 * submit — the user (or a real navigation) moved to a DIFFERENT chat — as
 * opposed to the programmatic churn a busy gateway produces constantly:
 *   - selection null-resets on a gateway/profile switch or reconnect
 *     (gateway-switch's `setSelectedStoredSessionId(null)`),
 *   - search/hash-only route changes from overlays and side panels,
 *   - background gateway events retargeting the active runtime id (#47709 class,
 *     which is why the active ref is not a prong here at all).
 * Returns null when nothing genuinely drifted, or a short reason string
 * (`route:<from>-><to>` / `selection:<from>-><to>`) for the abort log.
 */
export function sessionContextDrift({
  startRouteToken,
  nowRouteToken,
  startSelectedStoredId,
  nowSelectedStoredId,
  submitTargetStoredId
}: SessionContextDriftArgs): string | null {
  const targetStart = routeTargetFromToken(startRouteToken)
  const targetNow = routeTargetFromToken(nowRouteToken)

  // Route prong: the routed chat moved to a different, real chat. A null target
  // (navigated to settings / a non-chat overlay route) or a search/hash-only
  // change (same target) is not drift, and neither is landing on the submit's
  // own target.
  if (targetNow !== targetStart && targetNow !== null && targetNow !== submitTargetStoredId) {
    return `route:${targetStart}->${targetNow}`
  }

  // Selection prong: selection moved to a different, real stored session. A
  // null-reset (nowSelectedStoredId === null) or a move onto the submit's own
  // target is not drift.
  if (
    nowSelectedStoredId !== null &&
    nowSelectedStoredId !== startSelectedStoredId &&
    nowSelectedStoredId !== submitTargetStoredId
  ) {
    return `selection:${startSelectedStoredId}->${nowSelectedStoredId}`
  }

  return null
}
