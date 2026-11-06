import { host, randomUuid, useGatewayActivationEpoch } from '@hermes/plugin-sdk'
import { useEffect, useState } from 'react'

import { canonicalGroupRequest, canonicalGroupRoster } from './canonical-groups'
import type { CanonicalGroupBinding, CanonicalGroupRoute, CanonicalRoom, CanonicalRoomMember } from './canonical-groups'
import { uniqueGroupChatName } from './group-chat'
import { liveGroupChatNames } from './group-membership'
import { resolveBotConnectionRoute } from './routing'
import type { GroupMember, ProfileRoute, RosterRow } from './types'

/** Bots from other connections in a gateway Group Chat: Desktop runs the one-time setup (#130790)
 *  and keeps nothing. The room is created on this connection's gateway (the home) with the member
 *  pinned to its own gateway; `groups.peer.invite` there mints a grant for that room and member, and
 *  `groups.peer.register` on the home binds it. From then on the gateways run the member's turns.
 *  A setup that stops between its steps leaves a member that has not joined: `groups.state` shows
 *  it (`driver_status.peer_routes`), and running the steps again seats it. */

/** The one profile another gateway lends a room (gateway/session_group_peers.py `register`). */
const PEER_PROFILE = 'default'

/** The longest dispatch lifetime a member gateway grants, and, where it advertises
 *  `peer_renewal_horizon`, the longest horizon up to which the home may renew it. */
const LIFETIME = { ttl_seconds: 86400 }
const HORIZON = { status_ttl_seconds: 30 * 86400 }

/** Whether another connection's gateway can host a Bot in a room this connection owns, as its own
 *  `groups.capabilities` says. `gateway` is its authority id, the installation a pinned member names. */
export type PeerTargetState =
  | { ready: true; gateway: string; installationId: string; catalogDigest: string; renewal: boolean }
  | { ready: false; gateway?: string; reason: string }

type Answer = Record<string, unknown> & { room_link?: Record<string, any>; methods?: unknown; features?: unknown }

function peerCatalogProblem(catalog: Record<string, any>, gateway: string | undefined): string | undefined {
  const listed = (field: unknown, name: string | number) => Array.isArray(field) && field.includes(name)

  // What the home's registration requires of the member's gateway (`register`, `_probe`): the catalog
  // its endpoint answers with is the pinned one, it takes text only and lends its default profile.
  if (!listed(catalog.protocol_versions, 2) || !listed(catalog.link_modes, 'direct')) {return 'peer_target_unsupported'}

  if (catalog.persistent_process !== true) {return 'persistent_process_required'}

  if (catalog.text !== true || catalog.attachments !== false) {return 'text_only_required'}

  if (catalog.execution_policy?.target_profile !== PEER_PROFILE) {return 'default_profile_required'}

  if (catalog.endpoint?.available !== true || (typeof catalog.endpoint.url !== 'string' || !catalog.endpoint.url.trim())) {return 'endpoint_required'}

  if (!gateway || catalog.installation_id !== gateway || (typeof catalog.catalog_digest !== 'string' || !catalog.catalog_digest)) {return 'room_link_unavailable'}
}

export function peerTargetState(value: unknown): PeerTargetState {
  const answer: Answer = value && typeof value === 'object' ? value as Answer : {}
  const gateway = typeof answer.authority_gateway_id === 'string' && answer.authority_gateway_id ? answer.authority_gateway_id : undefined
  const listed = (field: unknown, name: string | number) => Array.isArray(field) && field.includes(name)
  const refused = (reason: string): PeerTargetState => ({ ready: false, reason, ...(gateway ? { gateway } : {}) })

  if (!listed(answer.methods, 'groups.peer.invite')) {return refused('peer_invite_unsupported')}
  const link = answer.room_link

  if (link?.enabled !== true) {return refused(typeof link?.reason === 'string' && link.reason ? link.reason : 'room_link_unavailable')}
  const catalog = link.catalog ?? {}

  const problem = peerCatalogProblem(catalog, gateway)

  if (problem || !gateway) {return refused(problem || 'room_link_unavailable')}

  return { ready: true, gateway, installationId: gateway, catalogDigest: catalog.catalog_digest, renewal: listed(answer.features, 'peer_renewal_horizon') }
}

/** A Bot's own connection route when it is a Bot another connection's room could seat: on another
 *  connection than `home`, consistently, and that connection's default profile. */
export function peerRoute(bot: Partial<RosterRow> | null | undefined, home: null | string): null | ProfileRoute {
  const { route } = resolveBotConnectionRoute(bot)

  if (bot?.ghost || !route || !home) {return null}

  return route.connectionId !== home && route.targetProfile === PEER_PROFILE &&
    (!bot?.connectionId || bot.connectionId === route.connectionId) ? route : null
}

function peerRequest<T>(route: ProfileRoute, method: string, params: Record<string, unknown> = {}): Promise<T> {
  return host.requestProfile<T>(route, method, { ...params, profile: route.targetProfile })
}

/** `groups.capabilities` of each connection the given Bots run on, while `active`: undefined while
 *  it is read, then that gateway's `peerTargetState`. Read again each time `active` turns on. */
export function usePeerTargets(bots: Partial<RosterRow>[], home: null | string, active: boolean, attempt = 0): ReadonlyMap<string, PeerTargetState> {
  const routes = new Map<string, ProfileRoute>()

  for (const bot of bots) {
    const route = peerRoute(bot, home)

    if (route && !routes.has(route.connectionId)) {routes.set(route.connectionId, route)}
  }

  const epoch = useGatewayActivationEpoch()
  const key = JSON.stringify([...routes.values()].map(route => [route.connectionId, route.mode, route.profile, route.targetProfile]).sort())
  const [read, setRead] = useState<{ key: string; epoch: number; states: ReadonlyMap<string, PeerTargetState> } | null>(null)

  useEffect(() => {
    if (!active) {return}
    let cancelled = false
    const states = new Map<string, PeerTargetState>()
    setRead({ key, epoch, states })

    for (const [connectionId, mode, profile, targetProfile] of JSON.parse(key) as [string, 'local' | 'remote', string, string][]) {
      void peerRequest<unknown>({ connectionId, mode, profile, targetProfile }, 'groups.capabilities')
        .then(peerTargetState, () => ({ ready: false, reason: 'unreachable' }) as const)
        .then(state => {
          if (cancelled) {return}
          states.set(connectionId, state)
          setRead({ key, epoch, states: new Map(states) })
        })
    }

    return () => { cancelled = true }
  }, [active, key, epoch, attempt])

  return read?.key === key && read.epoch === epoch ? read.states : new Map()
}

/** Fresh lender checks immediately before creating a room. */
export async function readPeerTargets(bots: RosterRow[], home: string): Promise<ReadonlyMap<string, PeerTargetState>> {
  const routes = new Map(bots.map(bot => peerRoute(bot, home)).filter((route): route is ProfileRoute => route !== null)
    .map(route => [route.connectionId, route]))

  return new Map(await Promise.all([...routes].map(async ([id, route]) => [id,
    await peerRequest(route, 'groups.capabilities').then(peerTargetState, () => ({ ready: false, reason: 'unreachable' }) as const)
  ] as const)))
}

/** A step Desktop can tell went wrong itself, beside the gateways' own refusals. */
export class PeerJoinError extends Error {
  constructor(readonly reason: 'peer_connection_missing' | 'peer_gateway_changed' | 'registration_unconfirmed' | string) {
    super(reason)
  }
}

interface PinnedPeer { kind: 'peer'; installation_id: string; capability_digest: string }
type RoomAuthority = Pick<CanonicalRoom, 'room_id' | 'authority_gateway_id' | 'authority_epoch'>
export interface PeerSeat { member: CanonicalRoomMember; route: ProfileRoute; renewal: boolean }

const pinned = (member: CanonicalRoomMember) => member.target?.kind === 'peer' ? member.target as unknown as PinnedPeer : null

/** Invite `seat.member` on its own gateway and register the grant on the home. The grant only
 *  passes through: if this stops halfway the member stays not joined and nothing here is owed. */
export async function seatPeer(home: CanonicalGroupBinding, room: RoomAuthority, seat: PeerSeat): Promise<void> {
  const pin = pinned(seat.member)

  if (!pin || !room.authority_gateway_id || typeof room.authority_epoch !== 'number') {throw new PeerJoinError('peer_gateway_changed')}

  const invitation = await peerRequest<{ grant?: unknown; target_profile?: unknown; catalog?: Record<string, any>; endpoint?: { url?: unknown } }>(
    seat.route, 'groups.peer.invite', {
      room_id: room.room_id, member_id: seat.member.member_id, home_install_id: room.authority_gateway_id,
      authority_gateway_id: room.authority_gateway_id, authority_epoch: room.authority_epoch, ...LIFETIME, ...(seat.renewal ? HORIZON : {})
    })

  // Only the gateway the room pinned the member to may answer for it, and the grant goes only to the
  // endpoint in that pinned catalog (the home refuses any other `target_url`).
  const endpoint = invitation?.catalog?.endpoint?.url

  if (typeof invitation?.grant !== 'string' || !invitation.grant || invitation.catalog?.installation_id !== pin.installation_id ||
    invitation.catalog.catalog_digest !== pin.capability_digest || typeof endpoint !== 'string' || invitation.endpoint?.url !== endpoint) {
    throw new PeerJoinError('peer_gateway_changed')
  }

  const receipt = await canonicalGroupRequest<{ registered?: unknown; target_install_id?: unknown }>(home, 'groups.peer.register', {
    room_id: room.room_id, member_id: seat.member.member_id, target_url: endpoint,
    target_profile: invitation.target_profile, grant: invitation.grant, catalog: invitation.catalog
  })

  if (receipt?.registered !== true || receipt.target_install_id !== pin.installation_id) {throw new PeerJoinError('registration_unconfirmed')}
}

async function seatPeers(home: CanonicalGroupBinding, room: RoomAuthority, seats: PeerSeat[]) {
  const unjoined: { member: CanonicalRoomMember; error: unknown }[] = []

  for (const seat of seats) {
    await seatPeer(home, room, seat).catch(error => { unjoined.push({ member: seat.member, error }) })
  }

  return unjoined
}

/** One creation attempt keeps its wire payload while the reply is uncertain. */
export interface GroupCreationIntent {
  key?: string
  payload?: { room_id: string; name: string; members: CanonicalRoomMember[] }
}

async function createGroupWithIntent(route: CanonicalGroupRoute, name: string, members: CanonicalRoomMember[],
  intent: GroupCreationIntent, key: string): Promise<CanonicalRoom> {
  if (intent.key !== key || !intent.payload) {
    intent.key = key
    intent.payload = { room_id: randomUuid(), name: uniqueGroupChatName(name, new Set(liveGroupChatNames())), members }
  }

  const { room } = await canonicalGroupRequest<{ room: CanonicalRoom }>(route, 'groups.create', intent.payload).catch(error => {
    // A coded refusal ends this attempt; a lost reply keeps the exact request for Retry.
    if (typeof error?.code === 'number') {delete intent.payload}
    throw error
  })

  delete intent.payload

  return room
}

/** Create a gateway room on `route` with Bots from other connections, then seat each of those.
 *  `targets` is `usePeerTargets`' answer: a Bot whose gateway is not ready is refused before anything
 *  is created. The room exists once this returns; `unjoined` names the members not seated yet. */
export async function createGroupAcrossConnections(
  route: CanonicalGroupRoute,
  name: string,
  members: GroupMember[],
  targets: ReadonlyMap<string, PeerTargetState>,
  intent: GroupCreationIntent
) {
  const seats = new Map<GroupMember, Omit<PeerSeat, 'member'>>()

  const checked = canonicalGroupRoster(route, members, member => {
    const own = peerRoute(member, route.connectionId)
    const target = own && targets.get(own.connectionId)

    if (!own || !target?.ready) {return undefined}
    seats.set(member, { route: own, renewal: target.renewal })

    return target
  })

  if (!name.trim() || 'problem' in checked) {
    throw Object.assign(new Error('A gateway group needs two to six unique members its gateway can seat'),
      { canonicalGroupProblem: 'problem' in checked ? checked.problem : 'groupSize' })
  }

  const key = JSON.stringify([route.connectionId, route.profile, name, members])

  const room = await createGroupWithIntent(route, name, checked.roster, intent, key)

  const binding: CanonicalGroupBinding = { ...route, roomId: room.room_id }

  const unjoined = await seatPeers(binding, room, checked.peers.map(({ member, source }) => ({ member, ...seats.get(source)! })))

  return { binding, room, unjoined }
}

interface PeerLabels {
  peerJoinFailed: string
  peerJoinUnreachable: string
  peerJoinChanged: string
}

const fill = (words: string, values: Record<string, string>) =>
  Object.entries(values).reduce((text, [key, value]) => text.replace(`{${key}}`, value), words)

/** Why `name` has not joined, in the reader's words: Desktop's own finding or the gateway's refusal
 *  (`message` is its reason on the gateway, its detail on `hermes serve`). */
export function peerJoinText(error: unknown, name: string, labels: PeerLabels): string {
  const refusal = error as { data?: { reason?: unknown; detail?: unknown }; message?: unknown; reason?: unknown } | null

  const reason = error instanceof PeerJoinError ? error.reason
    : String(refusal?.data?.reason || refusal?.message || error)

  if (reason === 'peer_unreachable') {return fill(labels.peerJoinUnreachable, { name })}

  if (['peer_gateway_changed', 'peer_target_mismatch', 'peer_target_unsupported'].includes(reason)) {return fill(labels.peerJoinChanged, { name })}
  const detail = typeof refusal?.data?.detail === 'string' && refusal.data.detail ? refusal.data.detail : reason

  return fill(labels.peerJoinFailed, { name, reason: detail })
}

