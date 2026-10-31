import { profileVisibleName } from '@hermes/plugin-sdk'

import { canonicalGroupRequest } from './canonical-groups'
import type { CanonicalGroupRoute, CanonicalRoom, CanonicalRoomMember } from './canonical-groups'
import type { RosterRow } from './types'

function localProfile(member: CanonicalRoomMember): string | undefined {
  if (!member.target) {return member.profile}

  if (member.target.kind !== 'local') {return undefined}

  return typeof member.target.profile === 'string' ? member.target.profile : member.profile
}

/** Read-only, room-scoped presentation. Never join the foreground's union roster by bare name. */
export async function projectCanonicalGroupNames(route: CanonicalGroupRoute, room: CanonicalRoom): Promise<CanonicalRoom> {
  if (!room.members?.some(member => localProfile(member) !== undefined)) {return room}

  // Display metadata is optional on older/disconnected gateways. Losing this read must not
  // break history, approvals or sending; the stored room and actor snapshots remain usable.
  const snapshot = await canonicalGroupRequest<{ profiles?: RosterRow[] }>(route, 'profiles.list', { include_sessions: false })
    .catch(() => undefined)

  const profiles = new Map((snapshot?.profiles ?? []).map(profile => [profile.name, profile]))

  const members = room.members.map(member => {
    const profile = profiles.get(localProfile(member) ?? '')

    if (!profile) {return member}
    // An identity-only profiles.list row is not evidence of a new primary name.
    // Suppress the helper's ID fallback so older gateways keep readable stored snapshots.
    const display_name = profileVisibleName({ ...(profile as { ui_meta?: unknown; display_name?: string }), name: '' })

    if (!display_name || display_name === profile.name) {return member}

    return { ...member, display_name, current_display_name: display_name }
  })

  return { ...room, members }
}
