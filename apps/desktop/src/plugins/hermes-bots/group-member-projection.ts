import { botRosterKey } from './data'
import type { GroupChat, GroupMember } from './types'

interface MemberProjection {
  hosted?: null | string
  members?: GroupMember[]
}

/** Hosted mirrors are whole cached rosters, not independent membership votes.
 * A mixed-version tie prefers the identity-bearing copy without trying to
 * equate foreign installations by a shared profile or display name. */
export function mergeProjectedMemberLists(remote?: MemberProjection, local?: MemberProjection): GroupMember[] {
  if (remote?.hosted || local?.hosted) {
    const lists = [local?.members, remote?.members].filter((list): list is GroupMember[] => Array.isArray(list))

    return lists.find(list => list.length > 0 && list.every(member => member?.hostedIdentity)) || lists[0] || []
  }

  const members = new Map<string, GroupMember>()

  for (const member of [
    ...(Array.isArray(remote?.members) ? remote.members : []),
    ...(Array.isArray(local?.members) ? local.members : [])
  ]) {
    members.set(botRosterKey(member), member)
  }

  return [...members.values()]
}

function membershipSignature(members: GroupMember[]): string {
  return JSON.stringify(
    members.map(member => [
      member?.hostedIdentity?.roomId,
      member?.hostedIdentity?.memberId,
      member?.hostedIdentity?.installationId,
      member?.hostedIdentity?.profile,
      member?.handle || member?.name,
      member?.title || '',
      member?.display_name || ''
    ])
  )
}

/** Only the existing rich record can carry local verification. Nothing from
 * ui_meta, including a forged verification flag, establishes authority. */
export function mergeMemberProjectionIntoRoom(
  existing: Partial<GroupChat>,
  projected: MemberProjection,
  preserve: boolean,
  newer: boolean
): { members: GroupMember[]; needsRefresh: boolean } {
  const current = Array.isArray(existing.members) ? existing.members : []
  const incoming = Array.isArray(projected.members) ? projected.members : []

  if (existing.hosted && existing.hostedMembersVerified) {
    return { members: current, needsRefresh: membershipSignature(current) !== membershipSignature(incoming) }
  }

  if (preserve) {
    return { members: current, needsRefresh: false }
  }

  const remote = incoming.map(member => ({ ...member, remoteSource: true }))
  const incomingProjection = { ...projected, members: remote }

  return {
    members: newer
      ? remote
      : existing.hosted || projected.hosted
        ? mergeProjectedMemberLists(incomingProjection, existing)
        : mergeProjectedMemberLists(existing, incomingProjection),
    needsRefresh: false
  }
}
