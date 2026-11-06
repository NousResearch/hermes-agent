import { atom, host, randomUuid, useGatewayActivationEpoch } from '@hermes/plugin-sdk'
import { useEffect, useState } from 'react'

import type { CanonicalGroupMessages } from './canonical-group-locales'
import { registerCanonicalGroup } from './canonical-group-registry'
import { canonicalGroupRequest, canonicalGroupRoster, createRoomWithIntent, disbandCanonicalGroup } from './canonical-groups'
import type {
  CanonicalGroupBinding, CanonicalGroupRoute, CanonicalRoom, CanonicalRoomMember, GroupCreationIntent
} from './canonical-groups'
import { uniqueGroupChatName } from './group-chat'
import { liveGroupChatNames } from './group-membership'
import { resolveBotConnectionRoute } from './routing'
import type { GroupMember, ProfileRoute, RosterRow } from './types'

/** Bots from other connections in a gateway Group Chat: Desktop runs the one-time setup (#130790)
 *  and keeps no grant. The room is created on this connection's gateway (the home) with the member
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

/** Why the last attempt to seat a member failed, per room and member, until one succeeds. In memory
 *  only: after a restart the room still shows the member as not joined, without the reason. */
export const $peerJoinErrors = atom<Record<string, unknown>>({})

export const peerJoinKey = (binding: CanonicalGroupBinding, memberId: string) =>
  JSON.stringify([binding.connectionId, binding.profile, binding.roomId, memberId])

function noteJoin(binding: CanonicalGroupBinding, memberId: string, error?: unknown) {
  const { [peerJoinKey(binding, memberId)]: _previous, ...rest } = $peerJoinErrors.get()
  $peerJoinErrors.set(error === undefined ? rest : { ...rest, [peerJoinKey(binding, memberId)]: error })
}

/** A step Desktop can tell went wrong itself, beside the gateways' own refusals. */
export class PeerJoinError extends Error {
  constructor(readonly reason: 'peer_connection_missing' | 'peer_gateway_changed' | 'registration_unconfirmed' | string) {
    super(reason)
  }
}

interface PinnedPeer { kind: 'peer'; installation_id: string; capability_digest: string }
type RoomAuthority = Pick<CanonicalRoom, 'room_id' | 'authority_gateway_id' | 'authority_epoch'>
export interface PeerSeat { member: CanonicalRoomMember; route: ProfileRoute; renewal: boolean; alreadyJoined?: boolean }

const pinned = (member: CanonicalRoomMember) => member.target?.kind === 'peer' ? member.target as unknown as PinnedPeer : null

/** Invite `seat.member` on its own gateway and register the grant on the home. The grant only
 *  passes through: if this stops halfway the member stays not joined and nothing here is owed. */
export async function seatPeer(home: CanonicalGroupBinding, room: RoomAuthority, seat: PeerSeat): Promise<void> {
  try {
    await invitedAndRegistered(home, room, seat)
    seatedAt.set(peerJoinKey(home, seat.member.member_id), Date.now())
    noteJoin(home, seat.member.member_id)
  } catch (error) {
    noteJoin(home, seat.member.member_id, error)
    throw error
  }
}

async function invitedAndRegistered(home: CanonicalGroupBinding, room: RoomAuthority, seat: PeerSeat) {
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

  await canonicalGroupRequest<{ registered?: unknown; target_install_id?: unknown }>(home, 'groups.peer.register', {
    room_id: room.room_id, member_id: seat.member.member_id, target_url: endpoint,
    target_profile: invitation.target_profile, grant: invitation.grant, catalog: invitation.catalog
  }).then(receipt => {
    if (receipt?.registered !== true || receipt.target_install_id !== pin.installation_id) {throw new PeerJoinError('registration_unconfirmed')}
  }).catch(async error => {
    const state = await canonicalGroupRequest<{ room?: CanonicalRoom; driver_status?: { peer_routes?: unknown } }>(home, 'groups.state', {
      room_id: room.room_id, include_disbanded: true
    }).catch(() => null)

    const sameRoom = state?.room?.room_id === room.room_id && state.room.authority_gateway_id === room.authority_gateway_id &&
      state.room.authority_epoch === room.authority_epoch

    // The room ended before this grant was registered: revoke it at its gateway, so no authority
    // outlives the room. (For a live room the scope also holds its working grant: keep it.)
    if (sameRoom && state.room?.disbanded_at != null) {
      await peerRequest(seat.route, 'groups.peer.revoke', { grant: invitation.grant }).catch(() => undefined)
      throw error
    }

    if (typeof error?.code !== 'number') {
      // Readiness can confirm a first join, but an old route cannot prove a fresh grant.
      if (!seat.alreadyJoined && sameRoom && peerMemberStates([seat.member], state?.driver_status?.peer_routes)[0]?.status === 'ready') {return}
      throw new PeerJoinError('registration_unconfirmed')
    }

    throw error
  })
}

async function seatPeers(home: CanonicalGroupBinding, room: RoomAuthority, seats: PeerSeat[]) {
  const unjoined: { member: CanonicalRoomMember; error: unknown }[] = []

  for (const seat of seats) {
    await seatPeer(home, room, seat).catch(error => { unjoined.push({ member: seat.member, error }) })
  }

  return unjoined
}

/** One creation attempt keeps its wire payload while the reply is uncertain. */
function createGroupWithIntent(route: CanonicalGroupRoute, name: string, members: CanonicalRoomMember[],
  intent: GroupCreationIntent, key: string): Promise<CanonicalRoom> {
  return createRoomWithIntent(route, intent, key, () => ({
    room_id: randomUuid(), name: uniqueGroupChatName(name, new Set(liveGroupChatNames())), members
  }))
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

/** How long a usable answer waits for another connection to that computer that may grant more. */
const ROUTE_GRACE_MS = 2_000

/** How well a connection's answer serves a member pinned to `installation` and `digest`: 0 another
 *  gateway or no answer; 1 that gateway, unable to host; 2 ready with
 *  another catalog; 3 ready with the pinned one; 4 that, granting the renewal horizon too. */
function routeRank(state: PeerTargetState | null, installation: string, digest: string) {
  if (state?.gateway !== installation) {return 0}

  return !state.ready ? 1 : state.catalogDigest !== digest ? 2 : state.renewal ? 4 : 3
}

/** The best route (`routeRank`) whose gateway is `installation`, asking every route at once, so a
 *  connection that never answers holds up none of the others. The best kind wins as soon as it
 *  answers, and a usable answer gives it `ROUTE_GRACE_MS` to do so; otherwise the best answer once
 *  all have answered, so another connection to the same computer that can't host it, or a shorter
 *  grant, never wins needlessly; null when none matches. */
function routeTo(installation: string, digest: string, routes: ProfileRoute[]) {
  return new Promise<{ route: ProfileRoute; state: PeerTargetState } | null>(resolve => {
    let left = routes.length
    let best: { route: ProfileRoute; state: PeerTargetState; rank: number } | null = null
    let grace: ReturnType<typeof setTimeout> | undefined

    const finish = () => {
      clearTimeout(grace)
      resolve(best)
    }

    if (!left) {resolve(null)}

    for (const route of routes) {
      void peerRequest<unknown>(route, 'groups.capabilities').then(peerTargetState, () => null).then(state => {
        const rank = routeRank(state, installation, digest)

        if (state && rank > (best?.rank ?? 0)) {best = { route, state, rank }}

        if (best?.rank === 4 || --left === 0) {
          finish()
        } else if (best?.rank === 3) {
          grace ??= setTimeout(finish, ROUTE_GRACE_MS)
        }
      })
    }
  })
}

/** The connection this Desktop reaches `member`'s gateway on: among the roster's Bots on other
 *  connections, the one whose gateway is the installation the room pinned the member to. */
export async function findPeerSeat(home: CanonicalGroupRoute, member: CanonicalRoomMember, roster: Partial<RosterRow>[]): Promise<PeerSeat> {
  const pin = pinned(member)
  const routes = new Map<string, ProfileRoute>()

  for (const bot of roster) {
    const route = peerRoute(bot, home.connectionId)

    if (route && !routes.has(route.connectionId)) {routes.set(route.connectionId, route)}
  }

  const found = pin && await routeTo(pin.installation_id, pin.capability_digest, [...routes.values()])

  if (!pin || !found) {throw new PeerJoinError('peer_connection_missing')}

  if (!found.state.ready) {throw new PeerJoinError(found.state.reason)}

  // Its gateway lends a different profile or policy now: this room's pin can never match it.
  if (found.state.catalogDigest !== pin.capability_digest) {throw new PeerJoinError('peer_gateway_changed')}

  return { member, route: found.route, renewal: found.state.renewal }
}

/** Check the room before inviting, and distinguish a first join from renewed access. */
async function peerAlreadyJoined(home: CanonicalGroupBinding, room: RoomAuthority, member: CanonicalRoomMember) {
  const answer = await canonicalGroupRequest<{ room?: Partial<CanonicalRoom>; driver_status?: { peer_routes?: unknown } }>(home, 'groups.state', {
    room_id: room.room_id, include_disbanded: true
  })

  if (answer?.room?.disbanded_at != null || answer?.room?.authority_gateway_id !== room.authority_gateway_id ||
    answer?.room?.authority_epoch !== room.authority_epoch) {throw new PeerJoinError('room_ended')}

  return peerMemberStates([member], answer.driver_status?.peer_routes)[0]?.status !== 'missing'
}

/** Seat a member of an existing room again (Retry): find its connection, invite, register. */
export async function rejoinPeer(binding: CanonicalGroupBinding, room: RoomAuthority & { members?: CanonicalRoomMember[] },
  member: CanonicalRoomMember, roster: Partial<RosterRow>[]) {
  const seat = await findPeerSeat(binding, member, roster).then(async found => {
    // Finding the member's gateway can take a while: never ask it for a grant for a room that ended.
    return { ...found, alreadyJoined: await peerAlreadyJoined(binding, room, member) }
  }).catch(error => {
    noteJoin(binding, member.member_id, error)
    throw error
  })

  await seatPeer(binding, room, seat)
}

/** Replace a frozen roster only after every remaining peer has joined the new room. */
export async function startAgainWithout(binding: CanonicalGroupBinding, room: CanonicalRoom, memberId: string,
  roster: Partial<RosterRow>[], labels: CanonicalGroupMessages, intent: GroupCreationIntent) {
  const members = room.members.filter(member => member.member_id !== memberId)
    .map(({ member_id, profile, handle, display_name, target }) => ({ member_id, profile, handle, target, ...(display_name ? { display_name } : {}) }))

  const name = intent.payload?.name ?? uniqueGroupChatName(room.name, new Set([room.name, ...liveGroupChatNames()]))
  const next = await createGroupWithIntent(binding, name, members, intent, peerJoinKey(binding, memberId))

  registerCanonicalGroup(binding, next, { created: true })
  const replacement: CanonicalGroupBinding = { connectionId: binding.connectionId, profile: binding.profile, roomId: next.room_id }
  const unjoined: { member: CanonicalRoomMember; error: unknown }[] = []

  for (const member of next.members.filter(pinned)) {
    await rejoinPeer(replacement, next, member, roster).catch(error => { unjoined.push({ member, error }) })
  }

  // An unreachable member keeps the original beside its replacement; retain the reason for the UI.
  let disbandError: unknown

  const ended = !unjoined.length && await disbandCanonicalGroup(binding, labels, labels.peerStartWithoutKept).catch(error => {
    disbandError = error

    return false
  })

  return { binding: replacement, room: next, unjoined, ended, disbandError }
}

/** When this Desktop last seated each member (`peerJoinKey`), in memory: after a restart, the first
 *  open of a room seats its members again. */
const seatedAt = new Map<string, number>()
/** Renewals in flight, per room (`roomKey`) and member. */
const renewing = new Map<string, Map<string, Promise<void>>>()
/** For a member whose last renewal found no connection to its gateway: the roster's other connections
 *  then. Its connection may just not have loaded, so a roster with another one tries it again at once. */
const missedWith = new Map<string, ReadonlySet<string>>()
const roomKey = (binding: CanonicalGroupBinding) => JSON.stringify([binding.connectionId, binding.profile, binding.roomId])

/** A seat made here stays fresh for less than its grant's dispatch lifetime (`ttl_seconds`, one
 *  day): until then the home can still renew the grant when it sends that member a turn. A failed
 *  renewal is tried again a few minutes after it failed. */
const SEAT_FRESH_MS = 20 * 60 * 60 * 1000
const RENEW_AGAIN_MS = 5 * 60 * 1000
/** How long a send waits for its room's renewals: past the home's probe of the member's gateway
 *  (30 s, `PeerRunsHTTPClient`) and the invitation before it. The send is journaled before it waits. */
const SEND_WAIT_MS = 45_000

/** Seat again each member of another gateway whose last seat here is not fresh, when its room is open
 *  and before a send (`peerRenewalSettled`). The home can renew a grant only while the grant still
 *  allows dispatch, which it does for one day, and only when it sends the member a turn; so without
 *  this, a member left a day without a turn needs a Retry. Only joined members, or members whose access
 *  ended, are renewed: one that never joined or is away keeps its recovery row. A roster with no
 *  Bot on another connection yet (still loading) renews nothing and counts as no attempt; one without
 *  the member's connection yet is tried again as soon as it has another. A renewal that fails changes
 *  nothing the member has now. */
export function renewPeers(binding: CanonicalGroupBinding, room: CanonicalRoom, routes: unknown, roster: Partial<RosterRow>[]) {
  const reach = new Set(roster.flatMap(bot => peerRoute(bot, binding.connectionId)?.connectionId ?? []))

  if (!reach.size) {return}
  const now = Date.now()
  const inRoom = renewing.get(roomKey(binding)) ?? new Map<string, Promise<void>>()
  renewing.set(roomKey(binding), inRoom)

  for (const { member, status } of peerMemberStates(room.members, routes)) {
    const key = peerJoinKey(binding, member.member_id)
    const missed = missedWith.get(key)
    const loaded = missed && [...reach].some(connection => !missed.has(connection))

    if ((status !== 'ready' && status !== 'needs_reauthorization') || inRoom.has(member.member_id) ||
      (now - (seatedAt.get(key) ?? 0) < SEAT_FRESH_MS && !loaded)) {continue}

    seatedAt.set(key, now)
    missedWith.delete(key)
    inRoom.set(member.member_id, rejoinPeer(binding, room, member, roster)
      .catch(error => {
        seatedAt.set(key, Date.now() - SEAT_FRESH_MS + RENEW_AGAIN_MS)

        if (error instanceof PeerJoinError && error.reason === 'peer_connection_missing') {missedWith.set(key, reach)}
      })
      .finally(() => inRoom.delete(member.member_id)))
  }
}

/** A send waits, for up to `timeoutMs`, for the renewals its room has in flight, so its turns go out
 *  under a fresh grant. */
export function peerRenewalSettled(binding: CanonicalGroupBinding, timeoutMs = SEND_WAIT_MS): Promise<void> {
  const inFlight = [...renewing.get(roomKey(binding))?.values() ?? []]

  if (!inFlight.length) {return Promise.resolve()}
  let timer: ReturnType<typeof setTimeout> | undefined

  return Promise.race([Promise.all(inFlight), new Promise(resolve => { timer = setTimeout(resolve, timeoutMs) })])
    .then(() => clearTimeout(timer))
}

/** A member the room pinned to another gateway, and whether it has joined (`ready`), as `groups.state`
 *  reports its route: absent until it registers, `needs_reauthorization` once its access ended,
 *  `unavailable` after a failed dispatch. The next dispatch checks availability again. */
export function peerMemberStates(members: CanonicalRoomMember[] = [], routes: unknown) {
  const statuses = new Map<string, string>()

  for (const route of Array.isArray(routes) ? routes : []) {
    if (route && typeof route.member_id === 'string' && typeof route.status === 'string') {statuses.set(route.member_id, route.status)}
  }

  return members.filter(member => pinned(member)).map(member => ({ member, status: statuses.get(member.member_id) ?? 'missing' }))
}

/** Retry can reseat a stored route only when this Desktop also reaches the pinned computer. */
export function canRejoinPeer(binding: CanonicalGroupBinding, member: CanonicalRoomMember, roster: Partial<RosterRow>[]): boolean {
  const pin = pinned(member)

  return !!pin && roster.some(bot => `install:${bot.installId}` === pin.installation_id &&
    bot.sourceReachable !== false && !bot.sourceMissing && !!peerRoute(bot, binding.connectionId))
}

type PeerLabels = Pick<CanonicalGroupMessages, 'peerJoinChanged' | 'peerJoinConnect' | 'peerJoinFailed' | 'peerJoinUnconfirmed' |
  'peerJoinUnreachable' | PeerSetupKey | 'peerSetupGeneric'>

const fill = (words: string, values: Record<string, string>) =>
  Object.entries(values).reduce((text, [key, value]) => text.replace(`{${key}}`, value), words)

type PeerSetupKey = 'peerSetupAddress' | 'peerSetupApiServer' | 'peerSetupApprovals' | 'peerSetupBackground' | 'peerSetupRoomKey' |
  'peerSetupRunStorage' | 'peerSetupUpdate'

/** What a lender's `groups.capabilities` refusal (`peerTargetState`: gateway/session_group_peers.py
 *  `room_link`, tui_gateway/methods_groups.py, or the catalog checks above) asks of its operator. */
const PEER_SETUP_TEXT: Record<string, PeerSetupKey> = {
  api_server_required: 'peerSetupApiServer',
  durable_run_storage_required: 'peerSetupRunStorage',
  gateway_roomlink_secret_unavailable: 'peerSetupRoomKey',
  execution_policy_unsupported: 'peerSetupApprovals',
  endpoint_required: 'peerSetupAddress',
  persistent_process_required: 'peerSetupBackground',
  peer_invite_unsupported: 'peerSetupUpdate',
  peer_target_unsupported: 'peerSetupUpdate',
  text_only_required: 'peerSetupUpdate',
  room_link_unavailable: 'peerSetupUpdate'
}

/** Why `connection` cannot lend a Bot to a room here, and what to change there; an unknown reason
 *  gets the general setup advice, never the raw code. */
export function peerSetupText(reason: string, connection: string, labels: Pick<CanonicalGroupMessages, PeerSetupKey | 'peerSetupGeneric' | 'peerTargetSetup'>): string {
  const key = Object.hasOwn(PEER_SETUP_TEXT, reason) ? PEER_SETUP_TEXT[reason] : 'peerSetupGeneric'

  return fill(labels.peerTargetSetup, { connection, reason: labels[key] })
}

/** Why `name` has not joined, in the reader's words: Desktop's own finding or the gateway's refusal
 *  (`message` is its reason on the gateway, its detail on `hermes serve`). */
export function peerJoinText(error: unknown, name: string, labels: PeerLabels): string {
  const refusal = error as { data?: { reason?: unknown; detail?: unknown }; message?: unknown; reason?: unknown } | null

  const reason = error instanceof PeerJoinError ? error.reason
    : String(refusal?.data?.reason || refusal?.message || error)

  if (reason === 'peer_connection_missing') {return fill(labels.peerJoinConnect, { name })}

  if (reason === 'peer_unreachable') {return fill(labels.peerJoinUnreachable, { name })}

  if (reason === 'registration_unconfirmed') {return fill(labels.peerJoinUnconfirmed, { name })}

  if (['peer_gateway_changed', 'peer_target_mismatch', 'peer_target_unsupported'].includes(reason)) {return fill(labels.peerJoinChanged, { name })}

  // A lender capability refusal (`findPeerSeat`) is a code; say what to change on its computer.
  if (Object.hasOwn(PEER_SETUP_TEXT, reason)) {return fill(labels.peerJoinFailed, { name, reason: labels[PEER_SETUP_TEXT[reason]] })}
  const detail = typeof refusal?.data?.detail === 'string' && refusal.data.detail ? refusal.data.detail : reason

  return fill(labels.peerJoinFailed, { name, reason: detail })
}
