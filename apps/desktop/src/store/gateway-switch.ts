import { atom } from 'nanostores'

import { resetLiveRuntimeTracking } from '@/app/contrib/hooks/use-background-sync'
import { resetSidebarBatchCapability } from '@/hermes'
import { invalidateProfileScopedQueries } from '@/lib/query-client'
import { clearArtifactRegistry } from '@/store/artifacts'
import { invalidateCronJobsRequests, setCronJobs } from '@/store/cron'
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
import { clearTranscriptTails } from '@/store/transcript-tail-cache'

// True while a connection switch is mid-flight — a Settings → Gateway apply
// (wipe → re-dial, use-gateway-boot softSwitch) or a Sessions-switcher source
// change (store/connections selectConnection). Lets the boot hook suppress the
// backend-exit toast, keeps the cold-boot CONNECTING overlay from resurrecting
// when startHermes re-emits boot progress, and tells the resume path that a
// "session not found" mid-switch means "retry once things settle", not "gone".
export const $gatewaySwitching = atom(false)

/**
 * Renderer-side cleanup a connection switch must run before the next gateway
 * is activated or published: fresh-draft the open transcript, drop the overlay
 * return route, reset the project tree, close terminals — everything bound to
 * the OUTGOING backend that lives outside the session store. Registered by
 * useGatewayBoot (whose host owns those React callbacks) so store-driven
 * switches run the exact same reset as a Settings → Gateway apply.
 */
export interface GatewaySwitchLifecycle {
  beforeConnectionSwitch: () => void
  /** Re-pull the session lists from whichever backend is active NOW. */
  refreshSessions: () => Promise<void>
}

let switchLifecycle: GatewaySwitchLifecycle | null = null

/** Ownership handle returned by beginGatewaySwitch; see endGatewaySwitch. */
export type GatewaySwitchToken = number

let latestSwitchToken = 0

export function registerGatewaySwitchLifecycle(lifecycle: GatewaySwitchLifecycle): () => void {
  switchLifecycle = lifecycle

  return () => {
    if (switchLifecycle === lifecycle) {
      switchLifecycle = null
    }
  }
}

/**
 * Commit point of every connection switch: raise the barrier and sever every
 * binding to the outgoing backend in ONE synchronous step. Both switch doors —
 * Settings apply (softSwitch) and the Sessions switcher (selectConnection) —
 * must call this BEFORE the next gateway is activated or its descriptor
 * published. The sidebar door used to activate first and wipe afterwards
 * (across an IPC round-trip), so route/session effects saw the new source while
 * $activeSessionId still named the previous backend's runtime and sent that id
 * to a backend that had never minted it — "session not found" (#93937).
 */
export function beginGatewaySwitch(): GatewaySwitchToken {
  const token = ++latestSwitchToken
  let wipeStarted = false

  $gatewaySwitching.set(true)

  try {
    switchLifecycle?.beforeConnectionSwitch()
    wipeStarted = true
    wipeSessionListsForGatewaySwitch()

    return token
  } catch (error) {
    // No caller received this token, so begin owns cleanup. Token-aware teardown
    // preserves a newer recursively-started switch, if lifecycle code began one.
    const stillOwnsSwitch = token === latestSwitchToken

    endGatewaySwitch(token)

    // A synchronous wipe has no rollback: once it starts, some outgoing-source
    // stores may already be empty. Repaint the still-active source best-effort.
    // A lifecycle failure happens before the wipe and leaves lists untouched.
    // If a nested switch superseded this one, its owner is responsible instead.
    if (wipeStarted && stillOwnsSwitch) {
      try {
        recoverActiveSourceAfterFailedGatewaySwitch(token)
      } catch {
        // Recovery must never replace the original commit failure.
      }
    }

    throw error
  }
}

/**
 * Lower the barrier once the switch that owns it has committed (or failed).
 * Switches overlap — a click can supersede one that is mid-commit — and the
 * barrier belongs to the LATEST one: an older switch ending is a no-op while a
 * newer one is still in flight. No token = force down (host teardown).
 */
export function endGatewaySwitch(token?: GatewaySwitchToken): void {
  if (token !== undefined && token !== latestSwitchToken) {
    return
  }

  $gatewaySwitching.set(false)
}

/**
 * A commit that fails AFTER beginGatewaySwitch leaves the still-active source
 * with its lists wiped and the sidebar skeleton armed, and nothing reactive
 * re-pulls them (no source/profile scope moved). Repaint it explicitly so the
 * sidebar doesn't sit on the skeleton; the fetch is best-effort. Recovery
 * retains the failed switch's token across the async refresh so it cannot
 * request through, or disarm loading for, a newer route.
 */
export function recoverActiveSourceAfterFailedGatewaySwitch(token: GatewaySwitchToken): void {
  const lifecycle = switchLifecycle

  if (!lifecycle) {
    console.debug('[gateway-switch] cannot repaint the active source because no switch lifecycle is registered')

    if (token === latestSwitchToken) {
      setSessionsLoading(false)
    }

    return
  }

  void Promise.resolve()
    .then(() => (token === latestSwitchToken ? lifecycle.refreshSessions() : undefined))
    .catch(() => undefined)
    .finally(() => {
      if (token === latestSwitchToken) {
        setSessionsLoading(false)
      }
    })
}

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
  invalidateCronJobsRequests()
  setCronJobs([])
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

  // Cached transcript tails belong to the PREVIOUS backend's sessions; a
  // different backend can recycle stored ids, and painting another machine's
  // conversation under a same-named id is worse than a loader. Wipe them.
  clearTranscriptTails()

  // Narrowed: account/marketplace/onboarding caches are global, not gateway-
  // scoped, so a mode swap must not refetch them.
  invalidateProfileScopedQueries()
}
