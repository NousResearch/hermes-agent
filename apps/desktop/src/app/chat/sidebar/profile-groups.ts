import { resolveProfileColor } from '@/lib/profile-color'
import { normalizeProfileKey } from '@/store/profile'
import type { SessionInfo } from '@/types/hermes'

import type { SidebarSessionGroup } from './projects'

/**
 * Build one collapsible group per profile over `sessions`, each carrying the
 * profile's color on the header. Default (root) profile floats to the top, the
 * rest sort alphabetically — the same ordering the recents list uses, so a
 * messaging platform's profile sub-groups line up with the recents groups.
 *
 * Shared by the recents list (profileGrouped view) and the messaging platform
 * sections (which group by profile only when the whole sidebar is in the
 * profile-grouped view), so both surfaces attribute rows to the same profile
 * key and paint them with the same color.
 */
export function buildProfileGroups(sessions: SessionInfo[], profileColors: Record<string, string>): SidebarSessionGroup[] {
  const groups = new Map<string, SidebarSessionGroup>()

  for (const session of sessions) {
    const key = normalizeProfileKey(session.profile)

    const group = groups.get(key) ?? {
      color: resolveProfileColor(key, profileColors),
      id: key,
      label: key,
      mode: 'profile',
      path: null,
      sessions: []
    }

    group.sessions.push(session)

    groups.set(key, group)
  }

  return [...groups.values()].sort((a, b) =>
    a.id === 'default' ? -1 : b.id === 'default' ? 1 : a.label.localeCompare(b.label)
  )
}
