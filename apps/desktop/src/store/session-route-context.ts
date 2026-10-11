/**
 * SESSION ROUTE CONTEXT — the generic "who/where is this conversation" answer
 * handed to per-session plugin decorations (sidebar rows, the composer's
 * session strip). Core owns lineage, ownership and routing; a plugin consumes
 * the result and never imports another feature's stores.
 */

import { LOCAL_CONNECTION_ID } from '@hermes/shared'
import { computed } from 'nanostores'

import type { SessionRouteContext } from '@/lib/session-row-slots'
import type { SessionInfo } from '@/types/hermes'

import { $activeConnectionId } from './connections'
import { $activeGatewayProfile, normalizeProfileKey } from './profile'
import { sessionMatchesStoredId, sessionPinId } from './session'
import { runtimeSessionOwner } from './session-states'

type RouteRow = Pick<SessionInfo, '_lineage_ids' | '_lineage_root_id' | 'connection_id' | 'id' | 'profile'>

/**
 * Row tags are the canonical route, so an UNTAGGED row is exactly "served by the primary pool".
 * Evidence (electron/profile-session-routing.ts `tagRegistrySessionResponse`; store/session.ts
 * `sessionOwnerRouteFromRow`, `stampMessagingRowsWithListServer`): every response from a
 * registry-pinned backend — including a REMOTE primary — is tagged with its registry id, and only
 * the primary pool (local, or the legacy primary-SSH path whose descriptor has no registry id) leaves
 * rows bare. `local` and "no id" therefore name the same primary pool, and a bare row is reachable
 * only while the active route is that pool. A bare row under an active non-primary registry
 * connection (`spark`) is NOT reachable: plugin REST would hit `spark`'s data. Fail closed.
 */
const connectionKey = (id: null | string | undefined): string => {
  const value = String(id ?? '').trim()

  return value === LOCAL_CONNECTION_ID ? '' : value
}

export interface ActiveRoute {
  connectionId: null | string
  profile: string
}

export function sessionRouteContext(session: RouteRow, active: ActiveRoute): SessionRouteContext {
  const connectionId = connectionKey(session.connection_id)
  const profile = normalizeProfileKey(session.profile)

  return {
    ambient: connectionId === connectionKey(active.connectionId) && profile === normalizeProfileKey(active.profile),
    connectionId,
    lineageIds: [
      ...new Set([session.id, session._lineage_root_id, ...(session._lineage_ids ?? [])].filter(Boolean) as string[])
    ],
    profile,
    sessionId: sessionPinId(session)
  }
}

/** The active route plugin REST is currently pinned to. */
export const $activeRoute = computed(
  [$activeConnectionId, $activeGatewayProfile],
  (connectionId, profile): ActiveRoute => ({ connectionId, profile })
)

/**
 * Context for a RUNTIME session (the composer's identity): its stored id and
 * lineage come from the listed row when there is one; the owner comes from the
 * row, else from what the runtime's own events proved, else the active route.
 * A session with no stored id yet (fresh draft) has no durable conversation to
 * describe → null.
 */
export function runtimeRouteContext(
  storedSessionId: null | string | undefined,
  sessions: readonly SessionInfo[],
  active: ActiveRoute,
  runtimeId: null | string | undefined
): null | SessionRouteContext {
  const stored = String(storedSessionId ?? '').trim()

  if (!stored) {
    return null
  }

  const row = sessions.find(session => sessionMatchesStoredId(session, stored))
  const owner = runtimeSessionOwner(runtimeId)
  // A bare profile string (legacy pool) or nothing proves no connection.
  const route = owner && typeof owner === 'object' ? owner : null

  if (row) {
    // The runtime's own events proved an exact connection; a bare row (optimistic, not yet
    // re-tagged by a list refresh) must not lose it to the primary pool.
    const tagged = !row.connection_id?.trim() && route ? { ...row, connection_id: route.connectionId } : row

    return sessionRouteContext(tagged, active)
  }

  return sessionRouteContext(
    {
      connection_id: route?.connectionId,
      id: stored,
      profile: typeof owner === 'string' ? owner : (route?.profile ?? active.profile)
    },
    active
  )
}
