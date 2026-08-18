import { atom } from 'nanostores'

import { resetLiveRuntimeTracking } from '@/app/contrib/hooks/use-background-sync'
import { resetSidebarBatchCapability } from '@/hermes'
import { invalidateProfileScopedQueries } from '@/lib/query-client'
import { clearArtifactRegistry } from '@/store/artifacts'
import { resetSessionsLimit } from '@/store/layout'
import { resetLiveSync } from '@/store/live-sync'
import { invalidateProfileListFetches } from '@/store/profile'
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
  setSessionProfilesTruncated,
  setSessionProfilesUsage,
  setSessions,
  setSessionsLoading
} from '@/store/session'
import { resetSessionPinMirror } from '@/store/session-pin-sync'
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
  // Strand any in-flight /api/profiles fetch from the PREVIOUS backend. The
  // rail's $profiles cache is deliberately NOT wiped (an empty list flickers
  // the rail away), but a late response from the old backend must not
  // overwrite what the new backend reports — that stale write is how a
  // remote/Cloud connection apply made the profile rail vanish (#85731).
  invalidateProfileListFetches()
  // Pins are mirrored per-backend. The next gateway has its own state.db and
  // has never seen them, so drop the "already pushed" bookkeeping and let the
  // next reconcile re-assert the whole set against the new backend.
  resetSessionPinMirror()
  setSessions([])
  setSessionProfilesTruncated({})
  setSessionProfilesUsage({})
  setCronSessions([])
  setMessagingSessions([])
  setMessagingPlatformTotals({})
  setMessagingTruncated(false)
  // Clearing $sessionStates automatically clears $workingSessionIds and
  // $attentionSessionIds (computed) and $stalledSessionIds (owned beside it).
  // $unreadFinishedSessionIds is separate, so wipe it explicitly. Only the
  // transient paint layer is wiped: the persisted markers/watermarks in
  // session-unread.ts are keyed by durable session id and repaint the rows
  // that are still unread once the next gateway's lists load — so a profile
  // round-trip doesn't swallow green dots.
  clearAllSessionStates()
  resetLiveRuntimeTracking()
  resetLiveSync()
  $unreadFinishedSessionIds.set([])
  setSessionsLoading(true)
  resetSessionsLimit()

  setActiveSessionId(null)
  setSelectedStoredSessionId(null)
  setMessages([])
  setFreshDraftReady(true)

  // Artifacts are keyed by sessions on the previous backend, so both the
  // registry and any rail tab pointing into it go with them.
  clearArtifactRegistry()

  // Narrowed: account/marketplace/onboarding caches are global, not gateway-
  // scoped, so a mode swap must not refetch them.
  invalidateProfileScopedQueries()
}
