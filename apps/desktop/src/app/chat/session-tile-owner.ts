import { knownSessionOwner } from '@/store/session'
import type { SessionOwnerRoute, SessionOwnerScope } from '@/store/session-request-router'
import type { SessionTile } from '@/store/session-states'
import type { SessionInfo } from '@/types/hermes'

/**
 * The owner a session tile routes its own RPCs through — the tile's explicit
 * route first, then the session row's `(connection, profile)` tag, with
 * `knownSessionOwner` folding in the owner hint.
 *
 * A tile opened without an explicit route — a branch child, which
 * `openSessionTile` creates with no `workspaceScope` — has no tile route, so
 * the row/hint rung is the only thing keeping its model and composer RPCs on
 * the backend that owns the session instead of the ambient one.
 *
 * A bare profile string carries no connection and is not a usable route:
 * handing it to `requestForSessionProfile` would resolve it against whichever
 * connection is active, which is the bug this ladder exists to avoid.
 */
export function tileOwnerRoute(
  tiles: readonly SessionTile[],
  rows: readonly SessionInfo[],
  storedSessionId: string
): SessionOwnerRoute | undefined {
  const owner: SessionOwnerScope =
    tiles.find(tile => tile.storedSessionId === storedSessionId)?.ownerRoute ??
    knownSessionOwner(rows, storedSessionId)

  if (!owner || typeof owner !== 'object' || !owner.connectionId) {
    return undefined
  }

  return {
    connectionId: owner.connectionId,
    profile: owner.profile,
    ...(owner.targetProfile ? { targetProfile: owner.targetProfile } : {})
  }
}
