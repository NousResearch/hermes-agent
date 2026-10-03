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
    tiles.find(tile => tile.storedSessionId === storedSessionId)?.ownerRoute ?? knownSessionOwner(rows, storedSessionId)

  if (!owner || typeof owner !== 'object' || !owner.connectionId) {
    return undefined
  }

  return {
    connectionId: owner.connectionId,
    profile: owner.profile,
    ...(owner.targetProfile ? { targetProfile: owner.targetProfile } : {})
  }
}

/**
 * Stable scalar identity for a resolved tile owner route.
 *
 * `tileOwnerRoute` builds a fresh object on every call, so a component that
 * derives the route through `useMemo` over a whole `$sessions` array re-creates
 * it (and everything keyed on it) whenever the list is republished — which
 * happens on every `sessions.changed` tick even when this tile's own row never
 * moved (multi-profile / Bots setups tick every ~2s). Encoding the route as a
 * string lets components subscribe through `useStoreSelector`/`useStoresSelector`
 * instead: `Object.is` bails out when the owner fields are unchanged, so
 * unrelated list churn stops at the subscription instead of re-rendering the
 * tile's chat shell.
 */
export function ownerRouteKey(route: SessionOwnerRoute | undefined): string | null {
  if (!route) {
    return null
  }

  return `${route.connectionId}\u0000${route.profile}\u0000${route.targetProfile ?? ''}`
}

/** Inverse of `ownerRouteKey`; the two live together so the field order cannot drift. */
export function ownerRouteFromKey(key: string | null): SessionOwnerRoute | undefined {
  if (!key) {
    return undefined
  }

  const [connectionId, profile, targetProfile] = key.split('\u0000')

  return { connectionId, profile, ...(targetProfile ? { targetProfile } : {}) }
}
