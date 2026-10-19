import type { SessionInfo } from '@/hermes'
import { useStoreSelector } from '@/lib/use-session-slice'
import { $profiles, normalizeProfileKey, profileNameFor } from '@/store/profile'

/**
 * #89888: the display name of the profile that owns a session, for lists that
 * merge profiles — the point of Show all is knowing whose chat a row is, and a
 * title plus a model does not say it.
 *
 * The name comes from the live roster (Bot Mode title → display_name → slug);
 * until that roster lands — or for a profile it no longer lists — the owning
 * key names itself, since that key IS the profile's name.
 *
 * Null when the row should carry no name at all: the caller's flag is off, or
 * the default profile owns the row (a mark on every row that says "the normal
 * one" is noise).
 */
export function useOwningProfileName(session: SessionInfo, showProfileName: boolean): null | string {
  const key = normalizeProfileKey(session.profile)
  const profileKey = showProfileName && key !== 'default' ? key : null
  const rosterName = useStoreSelector($profiles, profiles => (profileKey ? profileNameFor(profiles, profileKey) : null))

  return profileKey ? (rosterName ?? profileKey) : null
}
