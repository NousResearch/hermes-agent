import { normalizeProfileKey } from '@/store/profile'
import type { SessionInfo } from '@/types/hermes'

export function filterSessionsByProfileScope(
  sessions: SessionInfo[],
  profileScope: string,
  showAllProfiles: boolean
): SessionInfo[] {
  if (showAllProfiles) {
    return sessions
  }

  const scope = normalizeProfileKey(profileScope)

  return sessions.filter(session => normalizeProfileKey(session.profile) === scope)
}
