import type { SessionInfo } from '@/types/hermes'

export function buildSessionByAnyId(
  visibleSessions: SessionInfo[],
  cronSessions: SessionInfo[],
  messagingSessions: SessionInfo[]
): Map<string, SessionInfo> {
  const map = new Map<string, SessionInfo>()

  // Non-recents can still be pinned, so index them before recents. Recents
  // go last and win direct id collisions with the separately rendered groups.
  for (const session of [...cronSessions, ...messagingSessions, ...visibleSessions]) {
    map.set(session.id, session)

    if (session._lineage_root_id && !map.has(session._lineage_root_id)) {
      map.set(session._lineage_root_id, session)
    }
  }

  return map
}
