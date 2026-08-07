import type { SessionInfo } from '@/types/hermes'

/**
 * Index sessions by every id a pin might be stored under.
 *
 * The sidebar fetches three independent slices — recents, cron, and messaging
 * — and renders the latter two in self-managed sections. Any of them can be
 * pinned, so all three must be indexed here or the Pinned section can't
 * resolve the pin to a row. A pinned session is also filtered out of its own
 * section, so failing to index it doesn't merely misplace the row: it removes
 * the session from the sidebar entirely.
 *
 * Each session is keyed under both its live id and its lineage root, so a pin
 * stored before an auto-compression still resolves to the live continuation
 * tip. Recents are indexed last and win a direct id collision.
 */
export function buildSessionByAnyId(
  visibleSessions: SessionInfo[],
  cronSessions: SessionInfo[],
  messagingSessions: SessionInfo[]
): Map<string, SessionInfo> {
  const map = new Map<string, SessionInfo>()

  for (const session of [...cronSessions, ...messagingSessions, ...visibleSessions]) {
    map.set(session.id, session)

    if (session._lineage_root_id && !map.has(session._lineage_root_id)) {
      map.set(session._lineage_root_id, session)
    }
  }

  return map
}
