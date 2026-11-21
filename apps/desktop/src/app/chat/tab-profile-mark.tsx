import { ProfileTag } from '@/app/chat/profile-tag'
import { normalizeProfileKey } from '@/store/profile'

/**
 * The owning profile's mark in a session tab, to the left of the tab's name.
 *
 * Only a non-default profile earns one: the default profile is the app's own
 * workspace, so marking every tab with it would add a glyph to single-profile
 * users and say nothing. A named profile is real routing information — which
 * backend, which keys, which memory — and the sidebar row already says it, so
 * a tab that names only the session can otherwise be read as belonging to the
 * workspace the user happens to be in.
 *
 * Sits in the tab's `tabLead` (before the status dot), so it inherits the ⌘-
 * number overlay and the strip's own geometry. Sized down from the sidebar's
 * chip: a tab is 11px text, a 16px mark there would out-weigh the name.
 */
export function TabProfileMark({ profile }: { profile: null | string | undefined }) {
  const key = normalizeProfileKey(profile)

  if (key === 'default') {
    return null
  }

  return <ProfileTag className="mr-1 size-3.5 rounded-[2px] text-[0.4375rem]" profile={key} />
}
