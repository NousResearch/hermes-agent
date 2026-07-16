import { atom } from 'nanostores'

import { routeSessionId } from '@/app/routes'
import { invalidateProfileScopedQueries } from '@/lib/query-client'
import { resetSessionsLimit } from '@/store/layout'
import { requestFreshSession } from '@/store/profile'
import {
  $unreadFinishedSessionIds,
  setActiveSessionId,
  setCronSessions,
  setFreshDraftReady,
  setMessages,
  setMessagingPlatformTotals,
  setMessagingSessions,
  setMessagingTruncated,
  setRememberedSessionId,
  setResumeExhaustedSessionId,
  setResumeFailedSessionId,
  setSelectedStoredSessionId,
  setSessionProfileTotals,
  setSessions,
  setSessionsLoading,
  setSessionsTotal
} from '@/store/session'
import { clearAllSessionStates } from '@/store/session-states'

// True while a soft gateway-mode apply is mid-flight (wipe → re-dial). Lets the
// boot hook suppress the backend-exit toast and keeps the cold-boot CONNECTING
// overlay from resurrecting when startHermes re-emits boot progress. Also
// blocks useRouteResume from re-opening a previous-gateway session id.
export const $gatewaySwitching = atom(false)

/** HashRouter path (`#/session-id` → `/session-id`), with a pathname fallback. */
function currentAppPathname(): string {
  if (typeof window === 'undefined') {
    return '/'
  }

  const hash = window.location.hash.replace(/^#/, '')

  if (hash) {
    return hash.startsWith('/') ? hash : `/${hash}`
  }

  return window.location.pathname || '/'
}

/**
 * Clear gateway-bound session UI so sidebar skeletons retrigger.
 *
 * Sessions live in nanostores (not React Query) — refreshSessions merges into
 * the existing list, so without an explicit wipe a soft switch would keep
 * painting the previous gateway's rows. RQ caches (settings/config/skills) are
 * invalidated separately; the live session list is this path.
 *
 * Chat session routes (`#/:sessionId`) are reset to a blank draft — those ids
 * belong to the previous backend and would 404-spam resume after reconnect.
 * Overlay routes (Settings / Command Center / …) keep their URL so a mid-edit
 * Gateway settings page is not closed by the soft switch.
 */
export function wipeSessionListsForGatewaySwitch(): void {
  setSessions([])
  setSessionsTotal(0)
  setSessionProfileTotals({})
  setCronSessions([])
  setMessagingSessions([])
  setMessagingPlatformTotals({})
  setMessagingTruncated(false)
  // Clearing $sessionStates automatically clears $workingSessionIds and
  // $attentionSessionIds (they're computed from it). $unreadFinishedSessionIds
  // is separate (transient, not computable) so wipe it explicitly.
  clearAllSessionStates()
  $unreadFinishedSessionIds.set([])
  setSessionsLoading(true)
  resetSessionsLimit()

  setActiveSessionId(null)
  setSelectedStoredSessionId(null)
  setMessages([])
  setFreshDraftReady(true)
  // Stop the bounded resume retry + "last session" boot restore from chasing
  // an id that only existed on the previous backend.
  setResumeFailedSessionId(null)
  setResumeExhaustedSessionId(null)
  setRememberedSessionId(null)

  // Narrowed: account/marketplace/onboarding caches are global, not gateway-
  // scoped, so a mode swap must not refetch them.
  invalidateProfileScopedQueries()

  if (routeSessionId(currentAppPathname())) {
    // Navigate via the shared fresh-session request so the chat controller
    // owns the route flip (HashRouter) instead of us reaching into history.
    requestFreshSession()
  }
}
