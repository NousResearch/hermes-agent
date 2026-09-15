import type { SessionInfo } from '@/types/hermes'

/**
 * THE canonical write path for tagging backend-returned session rows with the
 * registry connection that owns them (the read counterpart is
 * `sessionOwnerRouteFromRow` in store/session-request-router).
 *
 * Every other `connection_id` writer works from an EXACT captured owner route
 * (the optimistic row in upsertOptimisticSession, the cache patch in
 * use-session-actions/utils, the mergeSessionPage carry) — those are
 * authoritative and this helper must never clobber them, so a row that
 * already names an owner is returned untouched. Only untagged rows served by
 * an active NON-local source get stamped: the gateway's HTTP APIs correctly
 * know nothing about Desktop-local registry ids, and an untagged remote row
 * would let a later resume fall back to a same-named local profile
 * ("session not found" on turn two). `local` is never stamped — a bare local
 * row already routes correctly and a `local` tag would only pin it against
 * the fail-closed owner resolution for no benefit.
 */
export function stampRowsWithOwningConnection(
  sessions: SessionInfo[],
  connectionId: null | string | undefined
): SessionInfo[] {
  const owner = String(connectionId ?? '').trim()

  if (!owner || owner === 'local') {
    return sessions
  }

  return sessions.map(session => (session.connection_id?.trim() ? session : { ...session, connection_id: owner }))
}

/** A durable backfill target: which backend store to stamp, expressed in the
 *  same scope vocabulary every session API call uses. `connectionId === null`
 *  means the primary/local backend's own store. */
export interface LegacyOwnerBackfillScope {
  connectionId: null | string
  profile: null | string
}

export interface LegacyOwnerBackfillTopology {
  /** Registry topology present (published registry, or the modern bridge). */
  hasRegistryTopology: boolean
  /** Non-local registered connection ids from the registry snapshot. */
  registryConnectionIds: string[]
  /** The source that served this page of rows: a registered connection id,
   *  `'local'`/`null` for the primary pool, `undefined` when the caller
   *  cannot name the serving source at all. */
  servingConnectionId: null | string | undefined
}

/**
 * Decide whether enumeration under the CURRENT topology warrants the
 * one-shot legacy owner backfill (#94724), and against WHICH store — the
 * durable counterpart of `stampRowsWithOwningConnection`'s in-memory stamp.
 *
 * Pre-#95407 rows carry `profile_name = NULL` in state.db. The list
 * endpoints decorate outgoing rows with their serving profile, so the
 * Desktop cannot see which rows are legacy from a page — but the SERVER
 * knows exactly, and its backfill is idempotent and one-shot-per-row. The
 * desktop's job is only to pick the store whose owner is a single match:
 * the backend that serves an enumeration owns every row it serves.
 *
 * Fail-closed rules (returns null — rows stay NULL, and the read-only
 * stored-transcript path keeps their history reachable):
 *  - no registry topology (nothing is broken; ambient owns every session);
 *  - the serving source is a connection the registry does not know;
 *  - the serving source cannot be named AND more than one registered
 *    backend could own the store (multi-candidate — never guess).
 */
export function resolveLegacyOwnerBackfillScope(
  topology: LegacyOwnerBackfillTopology
): LegacyOwnerBackfillScope | null {
  if (!topology.hasRegistryTopology) {
    return null
  }

  const serving = topology.servingConnectionId

  if (serving === null || serving?.trim() === 'local') {
    // Primary pool rows live in the primary's own per-profile store — a
    // single known owner, stamped as that store's own serving profile.
    return { connectionId: null, profile: null }
  }

  const servingId = serving?.trim() ?? ''
  const registered = topology.registryConnectionIds.map(id => id.trim()).filter(id => id && id !== 'local')

  if (servingId) {
    // The backend that served the rows owns them. Only a REGISTERED source
    // is a durable owner; an unknown source id cannot be trusted to survive.
    return registered.includes(servingId) ? { connectionId: servingId, profile: null } : null
  }

  // Serving source unknown: single-match only when exactly one registered
  // backend exists. Two or more candidates would make the stamp a guess.
  return registered.length === 1 ? { connectionId: registered[0], profile: null } : null
}
