/** Translate server-owned membership without making Desktop connections its identity. */

import type { HostedRoomCapability } from './hosted-room-client'
import { isHostedRoomContinuityEligible } from './hosted-room-client'
import { resolveBotConnectionRoute } from './routing'
import type { GroupMember } from './types'

interface HostedRoomMembership {
  authority_gateway_id?: unknown
  room_id?: unknown
  members?: unknown
}

function record(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>) : null
}

export function withoutHostedMemberRoute(member: GroupMember): GroupMember {
  const { connectionId: _connectionId, route: _route, ...identity } = member

  return { ...identity, sourceMissing: true, sourceReachable: false, sourceScoped: true }
}

/** Full and compact descriptors share the routing resolver. A projected
 * connection ID alone cannot prove that today's connection is this Bot. */
export function resolveHostedMemberDescriptor(
  member: GroupMember,
  capabilities: Record<string, HostedRoomCapability>
): GroupMember {
  const identity = member.hostedIdentity
  const route = resolveBotConnectionRoute(member).route
  const capability = route && capabilities[route.connectionId]

  if (
    !identity?.installationId ||
    !route ||
    route.targetProfile !== identity.profile ||
    capability?.authorityId !== identity.installationId
  ) {
    return withoutHostedMemberRoute(member)
  }

  return {
    ...member,
    route,
    connectionId: route.connectionId,
    sourceMissing: false,
    sourceReachable: isHostedRoomContinuityEligible(capability)
  }
}

export function hostedMemberDescriptors(
  room: HostedRoomMembership,
  homeConnectionId: string,
  existingMembers: GroupMember[],
  capabilities: Record<string, HostedRoomCapability>,
  sourceLabel: (connectionId: string) => string
): GroupMember[] {
  return (Array.isArray(room?.members) ? room.members : []).map(raw => {
    const member = record(raw) || {}
    const profile = String(member.profile || member.member_id || 'default')
    const target = record(member.target)
    const peer = target?.kind === 'peer'

    const installationId = peer
      ? String(target.installation_id || target.peer_id || '')
      : String(room.authority_gateway_id || capabilities[homeConnectionId]?.authorityId || '')

    const roomId = String(room.room_id || '')
    const memberId = String(member.member_id || '')

    const prior = existingMembers.find(
      candidate =>
        candidate.hostedIdentity?.roomId === roomId &&
        candidate.hostedIdentity?.memberId === memberId &&
        candidate.hostedIdentity?.installationId === installationId &&
        candidate.hostedIdentity?.profile === profile
    )

    // Only a currently inventoried connection with the exact installation can
    // supply a route. A persisted connection id can belong to another machine.
    const connectionId = peer
      ? Object.entries(capabilities).find(
          ([, capability]) => installationId && capability.authorityId === installationId
        )?.[0] || ''
      : homeConnectionId

    return {
      name: profile,
      handle: String(member.handle || member.profile || 'hermes'),
      title: String(member.display_name || ''),
      display_name: String(member.display_name || ''),
      hostedIdentity: { installationId, profile, roomId, memberId },
      ...(connectionId
        ? {
            connectionId,
            connectionLabel: sourceLabel(connectionId),
            route: { connectionId, mode: 'remote' as const, profile, targetProfile: profile }
          }
        : {
            ...(prior?.connectionLabel ? { connectionLabel: prior.connectionLabel } : {}),
            sourceMissing: true
          }),
      remoteSource: true,
      sourceScoped: true,
      sourceReachable: Boolean(connectionId && isHostedRoomContinuityEligible(capabilities[connectionId])),
      targetProfile: profile
    }
  })
}
