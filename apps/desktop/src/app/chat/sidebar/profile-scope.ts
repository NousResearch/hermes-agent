import { ALL_PROFILES, normalizeProfileKey } from '@/store/profile'
import type { SessionInfo } from '@/types/hermes'

/**
 * Sessions visible in one sidebar profile scope.
 *
 * ALL (`__all__`) returns the caller's list unchanged — including the
 * single-profile case, where Grouping → Profile persists that sentinel
 * while grouped rendering stays off. Never filter against `__all__`.
 */
export function filterSessionsByProfileScope(sessions: SessionInfo[], profileScope: string): SessionInfo[] {
  if (profileScope === ALL_PROFILES) {
    return sessions
  }

  // Case folded here, at the comparison, not inside normalizeProfileKey: the
  // canonical key is also the tile/session identity, where case is significant
  // (store/session-states.test.ts). Folding absorbs 'Default' (a title-cased
  // display label) vs the backend's canonical 'default' — the same alias
  // hermes_cli.profiles.normalize_profile_name matches case-insensitively.
  const scope = normalizeProfileKey(profileScope).toLowerCase()

  return sessions.filter(session => normalizeProfileKey(session.profile).toLowerCase() === scope)
}
