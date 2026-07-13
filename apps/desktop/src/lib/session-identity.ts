import type { SessionInfo } from '@/types/hermes'

type SessionIdentity = Pick<SessionInfo, '_lineage_ids' | '_lineage_root_id' | 'id'>

/** True when an id belongs to this logical conversation.
 *
 * Auto-compression rotates the live session id. A turn may therefore remain
 * keyed by an intermediate segment while the sidebar already projects a newer
 * tip. Matching the complete lineage keeps selection and working indicators
 * attached to the one visible conversation row. */
export function sessionMatchesStoredId(session: SessionIdentity, id: null | string | undefined): boolean {
  if (!id) {
    return false
  }

  return session.id === id || session._lineage_root_id === id || Boolean(session._lineage_ids?.includes(id))
}

export function sessionMatchesAnyId(session: SessionIdentity, ids: ReadonlySet<string>): boolean {
  if (ids.has(session.id) || (session._lineage_root_id != null && ids.has(session._lineage_root_id))) {
    return true
  }

  return Boolean(session._lineage_ids?.some(id => ids.has(id)))
}

export function sessionIdentityIds(session: SessionIdentity): string[] {
  return [
    ...new Set([session.id, session._lineage_root_id, ...(session._lineage_ids ?? [])].filter(Boolean) as string[])
  ]
}
