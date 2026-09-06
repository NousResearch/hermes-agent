/** Register peer execution and the optional return path for owner messaging. */

import { classifyHostedRoomCapability, profileScopedRoomLinkEndpoint } from './hosted-room-client'
import type { HostedRoomCapability } from './hosted-room-client'
import type { HostedRoomProbe } from './hosted-room-runtime'
import { requestHostedConnection } from './hosted-room-transport'
import type { GroupMember, ProfileRoute } from './types'

interface AutonomousHostedRoomMember {
  displayName?: string
  handle: string
  member: GroupMember
  profile: string
}

export interface AutonomousHostedRoomCreateInput {
  members: AutonomousHostedRoomMember[]
  name: string
  probe: HostedRoomProbe
  roomId: string
}

type PeerRequest = (method: string, params: Record<string, unknown>) => Promise<unknown>

export interface PreparedHostedPeer {
  capability: HostedRoomCapability | undefined
  registration: Record<string, unknown>
  requestPeer: PeerRequest
}

interface PeerControlInput {
  homeRoute: ProfileRoute
  homeCapability: HostedRoomCapability
  peerCapability: HostedRoomCapability | undefined
  roomId: string
  authorityId: string
  authorityEpoch: number
  memberId: string
  targetProfile: string
  targetAuthority: string
  requestId: string
  requestPeer: PeerRequest
  assertCurrent?: () => void
}

function record(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>) : {}
}

export async function registerHostedPeerControl(input: PeerControlInput) {
  const { homeCapability, peerCapability, requestPeer, assertCurrent = () => undefined } = input

  if (!homeCapability.reciprocalRoomControl || !peerCapability?.reciprocalRoomControl) {
    return false
  }

  const failure = () => new Error('Messaging could not connect to this Bot. Reconnect it and try again.')

  const homeUrl = profileScopedRoomLinkEndpoint(
    homeCapability.roomLink?.endpoint,
    input.homeRoute.targetProfile || input.homeRoute.profile || 'default'
  )

  if (
    !homeUrl ||
    homeCapability.authorityId !== input.authorityId ||
    peerCapability.authorityId !== input.targetAuthority
  ) {
    throw failure()
  }

  const digest = await globalThis.crypto.subtle.digest('SHA-256', new TextEncoder().encode(input.requestId))
  const requestId = `desktop-control:${[...new Uint8Array(digest)].map(byte => byte.toString(16).padStart(2, '0')).join('')}`
  assertCurrent()

  const invitation = record(
    await requestHostedConnection(input.homeRoute, 'groups.control.invite', {
      room_id: input.roomId,
      member_id: input.memberId,
      caller_install_id: input.targetAuthority,
      request_id: requestId,
      reuse_existing: true
    })
  )

  assertCurrent()

  if (
    invitation.room_id !== input.roomId ||
    invitation.member_id !== input.memberId ||
    invitation.authority_gateway_id !== input.authorityId ||
    invitation.authority_epoch !== input.authorityEpoch ||
    invitation.home_url !== homeUrl ||
    typeof invitation.control_token !== 'string' ||
    !/^[A-Za-z0-9_-]{32,256}$/.test(invitation.control_token) ||
    typeof invitation.expires_at !== 'number' ||
    !Number.isFinite(invitation.expires_at) ||
    invitation.expires_at <= 0
  ) {
    throw failure()
  }

  const result = record(
    await requestPeer('groups.control.register', {
      room_id: input.roomId,
      member_id: input.memberId,
      authority_gateway_id: input.authorityId,
      authority_epoch: input.authorityEpoch,
      room_name: invitation.room_name,
      member_count: invitation.member_count,
      control_token: invitation.control_token,
      home_url: homeUrl,
      expires_at: invitation.expires_at,
      profile: input.targetProfile
    })
  )

  assertCurrent()

  if (result.registered !== true || result.room_id !== input.roomId) {
    throw failure()
  }

  return true
}

export type PeerControlRecoveryInput = Omit<
  PeerControlInput,
  'homeCapability' | 'peerCapability' | 'requestPeer' | 'assertCurrent'
> & {
  peerRoute: ProfileRoute
}

export async function verifyHostedPeerControlScope(input: PeerControlRecoveryInput): Promise<
  | 'gone'
  | {
      homeCapability: HostedRoomCapability
      peerCapability: HostedRoomCapability
    }
> {
  const homeCapability = classifyHostedRoomCapability(
    await requestHostedConnection(input.homeRoute, 'groups.capabilities'),
    {
      connectionId: input.homeRoute.connectionId
    }
  )

  const peerCapability = classifyHostedRoomCapability(
    await requestHostedConnection(input.peerRoute, 'groups.capabilities'),
    {
      connectionId: input.peerRoute.connectionId
    }
  )

  if (homeCapability.authorityId !== input.authorityId || peerCapability.authorityId !== input.targetAuthority) {
    throw new Error('Messaging connection identity changed.')
  }

  let state: Record<string, unknown>

  try {
    state = record(await requestHostedConnection(input.homeRoute, 'groups.state', { room_id: input.roomId }))
  } catch (error) {
    const code = record(error).code ?? record(record(error).error).code

    if (code === 4114) {
      return 'gone'
    }

    throw error
  }

  const room = record(state.room)

  if (room.room_id !== input.roomId || !Array.isArray(room.members)) {
    throw new Error('Group Chat identity could not be verified.')
  }

  const member = room.members.map(record).find(value => value.member_id === input.memberId)
  const target = record(member?.target)

  if (
    room.disbanded_at ||
    room.authority_gateway_id !== input.authorityId ||
    room.authority_epoch !== input.authorityEpoch ||
    !member ||
    target.kind !== 'peer' ||
    target.installation_id !== input.targetAuthority ||
    String(target.profile || member.profile || '') !== input.targetProfile
  ) {
    return 'gone'
  }

  return { homeCapability, peerCapability }
}

export async function recoverHostedPeerControl(input: PeerControlRecoveryInput): Promise<'complete' | 'gone'> {
  const scope = await verifyHostedPeerControlScope(input)

  if (scope === 'gone') {return scope}

  const registered = await registerHostedPeerControl({
    ...input,
    ...scope,
    requestPeer: (method, params) => requestHostedConnection(input.peerRoute, method, params)
  })

  if (!registered) {
    throw new Error('Messaging setup needs compatible gateways.')
  }

  return 'complete'
}

export async function registerHostedPeers(
  input: AutonomousHostedRoomCreateInput,
  created: { authorityId: string; authorityEpoch: number; connectionId: string },
  peers: PreparedHostedPeer[]
) {
  const homeRoute = input.probe.routes[created.connectionId]
  const homeCapability = input.probe.capabilities[created.connectionId]

  for (const peer of peers) {
    await requestHostedConnection(homeRoute, 'groups.peer.register', peer.registration)
    await registerHostedPeerControl({
      homeRoute,
      homeCapability,
      peerCapability: peer.capability,
      roomId: input.roomId,
      authorityId: created.authorityId,
      authorityEpoch: created.authorityEpoch,
      memberId: String(peer.registration.member_id),
      targetProfile: String(peer.registration.target_profile),
      targetAuthority: String(record(peer.registration.catalog).installation_id),
      requestId: `${input.roomId}:${peer.registration.member_id}`,
      requestPeer: peer.requestPeer
    })
  }
}
