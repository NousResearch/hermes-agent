import { host, randomUuid } from '@hermes/plugin-sdk'

import { canonicalApprovalPreview } from './canonical-group-approval'
import type { CanonicalGroupMessages } from './canonical-group-locales'
import type { GroupMember } from './types'

export interface CanonicalGroupRoute {
  connectionId: string
  profile: string
}

export interface CanonicalGroupBinding extends CanonicalGroupRoute {
  roomId: string
  roomName?: string
}

export interface CanonicalRoomMember {
  member_id: string
  profile: string
  handle: string
  display_name?: string
  target?: Record<string, unknown>
}

export interface CanonicalRoomContext {
  purpose: string
  notes: string
  /** Absent on runtimes without lead routing (they behave as `everyone`); `auto` lets a model call pick. */
  routing?: 'auto' | 'everyone' | 'lead'
  lead_member_id?: string
}

/** The backend's cap on shared notes (UTF-8 bytes after trimming; `validate_context`). */
export const MAX_ROOM_NOTES_BYTES = 4096

export const roomNotesBytes = (notes: string) => new TextEncoder().encode(notes.trim()).length

export interface CanonicalRoom {
  room_id: string
  name: string
  members: CanonicalRoomMember[]
  authority_gateway_id?: string
  authority_epoch?: number
  disbanded_at?: number | null
  /** Seconds; the last change to the room (`groups.list` orders by it). */
  updated_at?: number
}

export interface CanonicalPendingAction {
  kind: string
  member_id: string
  task_id: string
  execution_generation: number
  request_id?: string
  /** The driver's view of an approval: the choices it offers and what the work would do. */
  approval?: { request_id?: string; prompt_id?: string; command?: string; description?: string; choices?: string[] }
}

function requireRoute(route: CanonicalGroupRoute): void {
  if (!route.connectionId?.trim() || !route.profile?.trim()) {
    throw new Error('Opening this group chat requires its connection and profile')
  }
}

export async function canonicalGroupRequest<T>(
  route: CanonicalGroupRoute,
  method: string,
  params: Record<string, unknown> = {}
): Promise<T> {
  requireRoute(route)

  if (params.profile !== undefined && params.profile !== route.profile) {
    throw new Error('This group chat belongs to a different profile')
  }

  // The descriptor overload never falls back to the foreground gateway.
  return host.requestProfile<T>({
    connectionId: route.connectionId,
    profile: route.profile,
    targetProfile: route.profile,
    mode: route.connectionId === 'local' ? 'local' : 'remote'
  }, method, { ...params, profile: route.profile })
}

/** A resolved RPC alone does not prove this room ended; only its matching tombstone does. */
export async function disbandCanonicalGroup(binding: CanonicalGroupBinding, labels: CanonicalGroupMessages, keptMessage = labels.disbandKept): Promise<boolean> {
  const answer = await canonicalGroupRequest<{ tombstone?: { room_id?: unknown; disbanded_at?: unknown } }>(binding, 'groups.disband', {
    room_id: binding.roomId, cancel_id: randomUuid()
  }).catch(error => {
    // These two refusal reasons come from revoking access, after Disband's acknowledged Stop.
    // Other failures (including untyped errors) do not prove that work stopped.
    const reasons = new Map([
      ['peer_unreachable', labels.disbandPeerUnreachable],
      ['room_stop_pending', labels.disbandStopPending],
      ['member_home_unknown', labels.disbandMemberHomeUnknown],
      ['room_disbanding_retry', labels.disbandInProgress],
      ['peer_revoke_unsupported', labels.disbandRevokeUnsupported],
      ['runtime_coordination_required', labels.disbandServiceRequired]
    ])

    const reason = reasons.get(error?.data?.reason) ??
      (typeof error?.message === 'string' && error.message.includes('still stopping') ? labels.disbandStopPending : labels.disbandUnconfirmed)

    throw new Error(keptMessage.replace('{reason}', reason))
  })

  const tombstone = answer?.tombstone

  return tombstone?.room_id === binding.roomId && typeof tombstone.disbanded_at === 'number' && Number.isFinite(tombstone.disbanded_at)
}

export async function discoverCanonicalGroups(
  route: CanonicalGroupRoute
): Promise<{ driver: boolean; rooms: CanonicalRoom[] }> {
  const capabilities = await canonicalGroupRequest<{ driver: boolean }>(route, 'groups.capabilities')

  return { driver: capabilities.driver === true, rooms: await listCanonicalRooms(route) }
}

/** Every live room on `route`, all pages. */
export async function listCanonicalRooms(route: CanonicalGroupRoute): Promise<CanonicalRoom[]> {
  const rooms: CanonicalRoom[] = []
  let offset = 0

  for (;;) {
    const page = await canonicalGroupRequest<{ rooms: CanonicalRoom[]; next_offset: number | null }>(
      route, 'groups.list', { limit: 100, offset }
    )

    rooms.push(...page.rooms)

    if (page.next_offset === null) {break}

    if (!Number.isSafeInteger(page.next_offset) || page.next_offset <= offset) {
      throw new Error('Could not load the next group chats')
    }

    offset = page.next_offset
  }

  return rooms
}

/** Why a roster cannot form a gateway group; the UI owns the words for each. */
export type CanonicalRosterProblem = 'groupSize' | 'otherConnection' | 'duplicateMember'

/** Whether `member` lives on `route`'s connection: a gateway group seats only those. */
export function onCanonicalConnection(route: CanonicalGroupRoute, member: GroupMember): boolean {
  const connectionId = member.route?.connectionId ?? member.connectionId

  return !((connectionId !== undefined && connectionId !== route.connectionId) ||
    (member.connectionId !== undefined && member.connectionId !== route.connectionId) ||
    (connectionId === undefined && member.remoteSource))
}

/** The gateway a Bot from another connection joins through, as `groups.capabilities` there names
 *  it (canonical-group-peers.ts `peerTargetState`): the room pins the member to it. */
export interface CanonicalPeerPin {
  installationId: string
  catalogDigest: string
}

/** A member on another connection the room seats as a peer, and the Bot it came from. */
export interface CanonicalPeerSeat {
  member: CanonicalRoomMember
  source: GroupMember
}

/** A Bot's own connection name as a handle suffix (`hermes` on "Mac Mini" joins as `hermes-mac-mini`). */
const slug = (value: unknown) => String(value ?? '').toLowerCase().replace(/[^a-z0-9_-]+/g, '-').replace(/^-+|-+$/g, '')

/** A short, stable digest (FNV-1a) of a connection id: the last resort for telling two apart. */
function digest(value: string) {
  let hash = 0x811c9dc5

  for (const char of value) {hash = Math.imul(hash ^ char.codePointAt(0)!, 0x01000193) >>> 0}

  return `c${hash.toString(36)}`
}

/** What tells a Bot's connection apart from its rivals' (members with the same handle), in order:
 *  its name, its id, a digest of its id. A rung is used only when it is set and no rival has the same
 *  one, so connections named in any script, or whose names read alike, still get distinct handles. */
function connectionSuffix(member: GroupMember, rivals: GroupMember[]) {
  const ladder = (bot: GroupMember) => {
    const id = String(bot.route?.connectionId || bot.connectionId || '')

    return [slug(bot.connectionLabel), slug(id), digest(id)]
  }

  const mine = ladder(member)
  const rung = mine.findIndex((suffix, index) => suffix && !rivals.some(rival => ladder(rival)[index] === suffix))

  return mine[rung < 0 ? 2 : rung]
}

/** A Bot's own name: its title, else its name (a default profile reads "Hermes"). */
const botLabel = (member: GroupMember) => String(member.title || member.display_name || '').trim() ||
  (member.name.trim().toLowerCase() === 'default' ? 'Hermes' : member.name)

/** A peer's name in the room: its own, then the connection it runs on, so two computers' default
 *  Bots never read alike. */
function peerLabel(member: GroupMember) {
  return member.connectionLabel ? `${botLabel(member)} · ${member.connectionLabel}` : botLabel(member)
}

const baseHandle = (member: GroupMember) => member.handle ?? member.route?.targetProfile ?? member.targetProfile ?? member.name

/** One member as `groups.create` takes it: a Bot here as its own profile, a pinned peer as its
 *  gateway's default profile under its handle, or handle-connection when another member has that
 *  handle. */
function rosterEntry(member: GroupMember, members: GroupMember[], pin: CanonicalPeerPin | null, mixed: boolean): CanonicalRoomMember {
  const base = baseHandle(member)
  const rivals = members.filter(other => other !== member && baseHandle(other).toLowerCase() === base.toLowerCase())

  if (!pin) {
    const profile = member.route?.targetProfile ?? member.targetProfile ?? member.name
    // Saved on the gateway and read from any computer: "This device" would be wrong elsewhere.
    const display = member.display_name || (mixed && profile === 'default' && base !== 'hermes' ? botLabel(member) : undefined)

    return { member_id: profile, profile, handle: base, target: { kind: 'local', profile },
      ...(display ? { display_name: display } : {}) }
  }

  const handle = rivals.length ? `${base}-${connectionSuffix(member, rivals)}` : base

  return { member_id: handle, profile: 'default', handle, display_name: peerLabel(member), target: {
    kind: 'peer', peer_id: pin.installationId, installation_id: pin.installationId, profile: 'default', capability_digest: pin.catalogDigest } }
}

/** The members `groups.create` would receive, or why these Bots cannot form a gateway group. A Bot on
 *  another connection is seated only when `peer` pins it to its own gateway (`rosterEntry`). Local
 *  profiles, handles (with the reserved `@all` and `@everyone`) and member ids are unique. */
export function canonicalGroupRoster(
  route: CanonicalGroupRoute,
  members: GroupMember[],
  peer?: (member: GroupMember) => CanonicalPeerPin | undefined
): { roster: CanonicalRoomMember[]; peers: CanonicalPeerSeat[] } | { problem: CanonicalRosterProblem } {
  if (members.length < 2 || members.length > 6) {
    return { problem: 'groupSize' }
  }

  const seen = new Set(['handle:all', 'handle:everyone'])
  const roster: CanonicalRoomMember[] = []
  const peers: CanonicalPeerSeat[] = []
  const mixed = members.some(member => !onCanonicalConnection(route, member))

  for (const member of members) {
    const pin = onCanonicalConnection(route, member) ? null : peer?.(member)

    if (pin === undefined) {
      return { problem: 'otherConnection' }
    }

    const entry = rosterEntry(member, members, pin, mixed)
    const keys = [`handle:${entry.handle}`, `id:${entry.member_id}`, ...(pin ? [] : [`profile:${entry.profile}`])].map(key => key.toLowerCase())

    if (!entry.profile.trim() || !entry.handle.trim() || keys.some(key => seen.has(key))) {
      return { problem: 'duplicateMember' }
    }

    keys.forEach(key => seen.add(key))
    roster.push(entry)

    if (pin) {peers.push({ member: entry, source: member })}
  }

  return { roster, peers }
}

export async function createCanonicalGroup(
  route: CanonicalGroupRoute,
  name: string,
  members: GroupMember[]
): Promise<{ binding: CanonicalGroupBinding; room: CanonicalRoom }> {
  requireRoute(route)
  const checked = canonicalGroupRoster(route, members)

  if (!name.trim()) {
    throw new Error('A group chat needs a name')
  }

  if ('problem' in checked) {
    throw Object.assign(new Error('A group chat hosted on this gateway needs two to six different bots on this connection'),
      { canonicalGroupProblem: checked.problem })
  }

  const { room } = await canonicalGroupRequest<{ room: CanonicalRoom }>(route, 'groups.create', {
    room_id: randomUuid(), name, members: checked.roster
  })

  return { binding: { ...route, roomId: room.room_id }, room }
}

interface CreateErrorLabels {
  createRefused: string
  createGroupSize: string
  createOtherConnection: string
  createDuplicateMember: string
}

/** A failed create in the reader's words: a roster problem found here, or the gateway's refusal
 *  followed by its reason. Any coded answer is the gateway refusing: the session gateway names the
 *  member in `data.detail` (4001); `hermes serve`, which Desktop's own backend and SSH, URL and
 *  Cloud connections reach, puts the reason in the message (4110, or 4123/5111 for a missing worker
 *  or a failure). Rejections arrive as Errors or, across the IPC bridge, as plain objects. */
export function canonicalGroupCreateError(error: unknown, labels: CreateErrorLabels): string {
  const problem = (error as { canonicalGroupProblem?: unknown } | null)?.canonicalGroupProblem

  const problems: Record<CanonicalRosterProblem, string> = {
    groupSize: labels.createGroupSize, otherConnection: labels.createOtherConnection, duplicateMember: labels.createDuplicateMember
  }

  if (typeof problem === 'string' && Object.hasOwn(problems, problem)) {
    return problems[problem as CanonicalRosterProblem]
  }

  const refusal = error as { code?: unknown; data?: { detail?: unknown }; message?: unknown } | null
  const message = typeof refusal?.message === 'string' ? refusal.message : String(error)

  if (typeof refusal?.code === 'number') {
    const reason = typeof refusal.data?.detail === 'string' && refusal.data.detail ? refusal.data.detail : message

    return reason ? `${labels.createRefused} ${reason}` : labels.createRefused
  }

  return message
}

export async function actCanonicalGroup(
  binding: CanonicalGroupBinding,
  action: CanonicalPendingAction,
  choice?: 'once' | 'deny'
): Promise<Record<string, unknown>> {
  const methods: Record<string, string> = { retry: 'groups.retry', discard: 'groups.discard', approval: 'groups.approve' }
  const method = Object.hasOwn(methods, action.kind) ? methods[action.kind] : undefined

  if (!method || !binding.roomId || !action.member_id || !action.task_id ||
    !Number.isSafeInteger(action.execution_generation) || action.execution_generation < 1) {
    throw new Error('This group chat action is no longer available')
  }

  const params: Record<string, unknown> = {
    room_id: binding.roomId,
    member_id: action.member_id,
    task_id: action.task_id,
    execution_generation: action.execution_generation
  }

  if (action.kind === 'approval') {
    if (!action.request_id || (choice !== 'once' && choice !== 'deny') ||
      !Array.isArray(action.approval?.choices) || !action.approval.choices.includes(choice)) {
      throw new Error('Approval requires its exact request ID and a choice the gateway offered')
    }

    if (choice === 'once' && !canonicalApprovalPreview(action).reviewable) {
      throw new Error('Allow once requires a reviewable command or edit')
    }

    params.request_id = action.request_id
    params.choice = choice
  }

  return canonicalGroupRequest(binding, method, params)
}
