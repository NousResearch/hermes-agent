import { resolveProfileColor } from '@/lib/profile-color'
import { normalizeProfileKey } from '@/store/profile'
import type { SessionInfo } from '@/types/hermes'

import type { SidebarSessionGroup } from './projects/workspace-groups'

/**
 * Build the ALL-profiles sidebar groups: one collapsible group per profile,
 * color on the header (not on every row).
 *
 * Ordering answers "what is THIS bot doing?" passively (#89347): the focused
 * chat's profile floats to the top, so switching bots re-ranks the list and
 * the bot you are looking at heads it. `default` keeps its historical spot
 * right after (first when nothing else is focused), and the rest sort
 * alphabetically. A focused profile with no sessions simply has no group —
 * ranking never invents empty groups.
 */
export function buildProfileGroups(
  sessions: readonly SessionInfo[],
  profileColors: Record<string, string>,
  focusedProfile: null | string
): SidebarSessionGroup[] {
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

  const focused = focusedProfile ? normalizeProfileKey(focusedProfile) : null

  const rank = (group: SidebarSessionGroup) => (group.id === focused ? 0 : group.id === 'default' ? 1 : 2)

  return [...groups.values()].sort((a, b) => rank(a) - rank(b) || a.label.localeCompare(b.label))
}
