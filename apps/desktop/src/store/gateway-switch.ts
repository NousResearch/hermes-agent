import { atom } from 'nanostores'

import { queryClient } from '@/lib/query-client'
import { resetSessionsLimit } from '@/store/layout'
import {
  setActiveSessionId,
  setAttentionSessionIds,
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
  setSessionsTotal,
  setWorkingSessionIds
} from '@/store/session'

// True while a soft gateway-mode apply is mid-flight (wipe → re-dial). Lets the
// boot hook suppress the backend-exit toast and keeps the cold-boot CONNECTING
// overlay from resurrecting when startHermes re-emits boot progress.
export const $gatewaySwitching = atom(false)

const PREVIEW_HOLD_MS = 1400

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
  setSessions([])
  setSessionsTotal(0)
  setSessionProfileTotals({})
  setCronSessions([])
  setMessagingSessions([])
  setMessagingPlatformTotals({})
  setMessagingTruncated(false)
  setWorkingSessionIds([])
  setAttentionSessionIds([])
  setSessionsLoading(true)
  resetSessionsLimit()

  setActiveSessionId(null)
  setSelectedStoredSessionId(null)
  setMessages([])
  setFreshDraftReady(true)

  void queryClient.invalidateQueries()
}

/**
 * Dev review beat: wipe → skeletons for PREVIEW_HOLD_MS → clear loading.
 * Does not tear down a real backend. Fired from the Settings button (Electron
 * has no easy `?query=` entry).
 */
export async function previewGatewaySwitch(holdMs = PREVIEW_HOLD_MS): Promise<void> {
  if ($gatewaySwitching.get()) {
    return
  }

  $gatewaySwitching.set(true)
  wipeSessionListsForGatewaySwitch()

  try {
    await new Promise<void>(resolve => {
      window.setTimeout(resolve, holdMs)
    })
  } finally {
    setSessionsLoading(false)
    $gatewaySwitching.set(false)
  }
}
