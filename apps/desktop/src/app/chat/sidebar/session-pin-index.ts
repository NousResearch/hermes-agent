import type { SessionInfo } from '@/hermes'
import { normalizeProfileKey } from '@/store/profile'

export function buildPinnedSessionIndex(
  profileScope: string,
  aggregateAllProfiles: boolean,
  ...sessionGroups: SessionInfo[][]
): Map<string, SessionInfo> {
  const index = new Map<string, SessionInfo>()

  for (const session of sessionGroups.flat()) {
    if (!aggregateAllProfiles && normalizeProfileKey(session.profile) !== profileScope) {
      continue
    }

    index.set(session.id, session)

    if (session._lineage_root_id && !index.has(session._lineage_root_id)) {
      index.set(session._lineage_root_id, session)
    }
  }

  return index
}
