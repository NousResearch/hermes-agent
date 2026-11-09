import { useStore } from '@nanostores/react'

import { ProfileTag } from '@/app/chat/profile-tag'
import { $profiles, normalizeProfileKey } from '@/store/profile'

/**
 * The owning profile's mark for a session's TAB.
 *
 * In the tiled/tabbed layout every session — the workspace one included — is a
 * pane tile, and the pane header that carries {@link ProfileTag} in the
 * single-pane layout is `isPrimary`-only. So a tab names the chat but never
 * says whose agent answers it, which is exactly the ambiguity the header tag
 * was added for (#66003).
 *
 * Renders nothing on a single-profile gateway, and nothing for the default
 * profile: the badge answers "which agent", so "the normal one" stays the
 * absence of a mark — the same rule the sidebar row follows.
 */
export function TabOwnerBadge({ profile }: { profile: null | string | undefined }) {
  const profiles = useStore($profiles)
  const key = normalizeProfileKey(profile)

  if (profiles.length < 2 || key === 'default') {
    return null
  }

  return <ProfileTag profile={profile} />
}
