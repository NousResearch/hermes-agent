/**
 * DOCK BADGE — the number of sessions waiting on the owner, on the app's Dock
 * icon (macOS; Linux launchers that honour app.setBadgeCount too). The window
 * is usually behind something else when a clarify or approval prompt lands;
 * the badge is what still says "go back, it needs you".
 *
 * Counts LISTED rows, not $sessionDotStateById keys: a compressed lineage
 * claims its state under every alias (see `lineageAliases`), so one session
 * would otherwise count once per tip. Archived rows are not waiting on anyone.
 */

import { computed } from 'nanostores'

import { $messagingSessions, $sessions } from './session'
import { $sessionDotStateById, type SessionDotState, sessionStatusBucket } from './session-dot-state'

export function needsInputSessionCount(
  byId: Readonly<Record<string, SessionDotState>>,
  ...lists: Array<readonly { archived?: boolean; id: string }[]>
): number {
  const ids = new Set<string>()

  for (const rows of lists) {
    for (const row of rows) {
      if (!row.archived && sessionStatusBucket(byId[row.id]) === 'needs-input') {
        ids.add(row.id)
      }
    }
  }

  return ids.size
}

export const $needsInputSessionCount = computed(
  [$sessionDotStateById, $sessions, $messagingSessions],
  (byId, sessions, messaging) => needsInputSessionCount(byId, sessions, messaging)
)

/** Mirror the count onto the Dock icon; 0 clears it. Only the main window
 *  owns the badge (`?win=` marks every secondary/HUD/overlay window), so two
 *  windows never fight over it. Returns the unsubscribe. */
export function installDockBadge(
  setCount: ((count: number) => unknown) | undefined = window.hermesDesktop?.setDockBadgeCount,
  search = window.location.search
): () => void {
  if (!setCount || new URLSearchParams(search).has('win')) {
    return () => {}
  }

  return $needsInputSessionCount.subscribe(count => {
    void setCount(count)
  })
}
