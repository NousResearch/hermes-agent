/**
 * DB-live gateway reconciliation — the seam for reconcileLiveGateways, split
 * from gateway.ts (its file is at its line cap). gateway.ts owns the secondary
 * registry (openGatewayForProfile, pruneSecondaryGateways); this sibling only
 * wires the two halves for cross-process liveness (#85302).
 */
import { openGatewayForProfile, pruneSecondaryGateways } from './gateway'

/** Open sockets for every profile in `keep` that isn't open yet, then prune
 *  the rest. The open half breaks the old circularity: a profile whose row is
 *  DB-live (foreign liveness — a cron run, a CLI one-shot) gets a socket
 *  without any user intent, so its serve's stream events and active_list
 *  reach the renderer and its rows stay honest. Pruning still applies, so a
 *  profile whose last DB-fresh row ages out (300s window) releases its
 *  socket on the next recompute. */
export function reconcileLiveGateways(keep: Set<string>): void {
  for (const profile of keep) {
    void openGatewayForProfile(profile).catch(() => undefined)
  }

  pruneSecondaryGateways(keep)
}
