import { atom } from 'nanostores'

import { resetSidebarBatchCapability } from '@/hermes'
import { invalidateProfileScopedQueries } from '@/lib/query-client'
import { resetSessionsLimit } from '@/store/layout'
import {
  $unreadFinishedSessionIds,
  setActiveSessionId,
  setCronSessions,
  setFreshDraftReady,
  setMessages,
  setMessagingPlatformTotals,
  setMessagingSessions,
  setMessagingTruncated,
  setSelectedStoredSessionId,
  setSessionProfileTotals,
  setSessions,
  setSessionsLoading,
  setSessionsTotal
} from '@/store/session'
import { clearAllSessionStates } from '@/store/session-states'

// True while a soft gateway-mode apply is mid-flight (wipe → re-dial). Lets the
// boot hook suppress the backend-exit toast and keeps the cold-boot CONNECTING
// overlay from resurrecting when startHermes re-emits boot progress.
export const $gatewaySwitching = atom(false)

/**
 * Clear gateway-bound session UI so sidebar skeletons retrigger.
 *
 * Sessions live in nanostores (not React Query) — refreshSessions merges into
 * the existing list, so without an explicit wipe a soft switch would keep
 * painting the previous gateway's rows. RQ caches (settings/config/skills) are
 * invalidated separately; the live session list is this path.
 *
 * Does NOT call requestFreshSession() — that navigates to NEW_CHAT and would
 * close route overlays (Settings). Clear chat state in place; leave the URL
 * alone so the user stays where they were (e.g. mid-Gateway settings).
 */
export function wipeSessionListsForGatewaySwitch(): void {
  // The next backend is a different runtime — don't carry the old one's
  // "batched sidebar endpoint missing" capability verdict across the switch.
  resetSidebarBatchCapability()
  setSessions([])
  setSessionsTotal(0)
  setSessionProfileTotals({})
  setCronSessions([])
  setMessagingSessions([])
  setMessagingPlatformTotals({})
  setMessagingTruncated(false)
  // Clearing $sessionStates automatically clears $workingSessionIds and
  // $attentionSessionIds (computed) and $stalledSessionIds (owned beside it).
  // $unreadFinishedSessionIds is separate, so wipe it explicitly.
  clearAllSessionStates()
  $unreadFinishedSessionIds.set([])
  setSessionsLoading(true)
  resetSessionsLimit()

  setActiveSessionId(null)
  setSelectedStoredSessionId(null)
  setMessages([])
  setFreshDraftReady(true)

  // Narrowed: account/marketplace/onboarding caches are global, not gateway-
  // scoped, so a mode swap must not refetch them.
  invalidateProfileScopedQueries()
}
