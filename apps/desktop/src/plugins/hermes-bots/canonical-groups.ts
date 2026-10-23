import { host } from '@hermes/plugin-sdk'

import { canonicalApprovalPreview } from './canonical-group-approval'
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

export interface CanonicalRoom {
  room_id: string
  name: string
  members: CanonicalRoomMember[]
  disbanded_at?: number | null
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
    throw new Error('Canonical groups require an explicit connection and profile')
  }
}

export function captureCanonicalGroupRoute(): CanonicalGroupRoute {
  const connectionId = host.state.connectionId.get()

  if (connectionId === null) {throw new Error('Canonical groups require an explicit connection')}
  const route = { connectionId, profile: host.state.profile.get() }
  requireRoute(route)

  return route
}

export async function canonicalGroupRequest<T>(
  route: CanonicalGroupRoute,
  method: string,
  params: Record<string, unknown> = {}
): Promise<T> {
  requireRoute(route)

  if (params.profile !== undefined && params.profile !== route.profile) {
    throw new Error('Canonical group profile does not match its authority')
  }

  // The descriptor overload never falls back to the foreground gateway.
  return host.requestProfile<T>({
    connectionId: route.connectionId,
    profile: route.profile,
    targetProfile: route.profile,
    mode: route.connectionId === 'local' ? 'local' : 'remote'
  }, method, { ...params, profile: route.profile })
}

export async function discoverCanonicalGroups(
  route: CanonicalGroupRoute
): Promise<{ driver: boolean; rooms: CanonicalRoom[] }> {
  const capabilities = await canonicalGroupRequest<{ driver: boolean }>(route, 'groups.capabilities')
  const rooms: CanonicalRoom[] = []
  let offset = 0

  for (;;) {
    const page = await canonicalGroupRequest<{ rooms: CanonicalRoom[]; next_offset: number | null }>(
      route, 'groups.list', { limit: 100, offset }
    )

    rooms.push(...page.rooms)

    if (page.next_offset === null) {break}

    if (!Number.isSafeInteger(page.next_offset) || page.next_offset <= offset) {
      throw new Error('Invalid canonical group pagination cursor')
    }

    offset = page.next_offset
  }

  return { driver: capabilities.driver === true, rooms }
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

/** The members `groups.create` would receive, or why these Bots cannot form a gateway group. */
export function canonicalGroupRoster(
  route: CanonicalGroupRoute,
  members: GroupMember[]
): { roster: CanonicalRoomMember[] } | { problem: CanonicalRosterProblem } {
  if (members.length < 2 || members.length > 6) {
    return { problem: 'groupSize' }
  }

  const profiles = new Set<string>()
  const handles = new Set(['all', 'everyone'])
  const roster: CanonicalRoomMember[] = []

  for (const member of members) {
    if (!onCanonicalConnection(route, member)) {
      return { problem: 'otherConnection' }
    }

    const profile = member.route?.targetProfile ?? member.targetProfile ?? member.name
    const handle = member.handle ?? profile

    if (!profile.trim() || !handle.trim() || profiles.has(profile.toLowerCase()) || handles.has(handle.toLowerCase())) {
      return { problem: 'duplicateMember' }
    }

    profiles.add(profile.toLowerCase())
    handles.add(handle.toLowerCase())
    roster.push({
      member_id: profile, profile, handle,
      target: { kind: 'local', profile },
      ...(member.display_name ? { display_name: member.display_name } : {})
    })
  }

  return { roster }
}

export async function createCanonicalGroup(
  route: CanonicalGroupRoute,
  name: string,
  members: GroupMember[]
): Promise<{ binding: CanonicalGroupBinding; room: CanonicalRoom }> {
  requireRoute(route)
  const checked = canonicalGroupRoster(route, members)

  if (!name.trim()) {
    throw new Error('A canonical group needs a name')
  }

  if ('problem' in checked) {
    throw Object.assign(new Error('A canonical group needs two to six unique members on its authority connection'),
      { canonicalGroupProblem: checked.problem })
  }

  const { room } = await canonicalGroupRequest<{ room: CanonicalRoom }>(route, 'groups.create', {
    room_id: crypto.randomUUID(), name, members: checked.roster
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
    throw new Error('Invalid canonical group pending action')
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
