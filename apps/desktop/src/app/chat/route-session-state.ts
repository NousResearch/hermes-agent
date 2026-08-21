import { sessionMatchesStoredId } from '@/store/session'
import type { SessionInfo } from '@/types/hermes'

/**
 * Whether the route points at a different conversation than the selected view.
 *
 * Auto-compression rotates a conversation from its root id to a continuation
 * tip while the durable URL intentionally stays on the root. Those ids are
 * different strings but still the same conversation, so a loaded lineage row
 * must win over the raw comparison. An unknown route remains a mismatch: that
 * is the real navigation case where the old transcript/composer must hide
 * until resume finishes.
 */
export function isRouteSessionMismatch(
  routedSessionId: null | string,
  selectedSessionId: null | string,
  sessions: readonly Pick<SessionInfo, '_lineage_root_id' | 'id'>[]
): boolean {
  if (!routedSessionId || routedSessionId === selectedSessionId) {
    return false
  }

  if (!selectedSessionId) {
    return true
  }

  return !sessions.some(
    session => sessionMatchesStoredId(session, routedSessionId) && sessionMatchesStoredId(session, selectedSessionId)
  )
}
