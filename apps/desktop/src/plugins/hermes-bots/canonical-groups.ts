import { host } from '@hermes/plugin-sdk'

import { canonicalApprovalPreview } from './canonical-group-approval'
import { groupExecutionMode } from './canonical-group-capabilities'
import type { GroupExecutionMode } from './canonical-group-capabilities'
import { CANONICAL_GROUP_LOCALES } from './canonical-group-locales'
import type { GroupMember } from './types'

export interface CanonicalGroupRoute {
  connectionId: string
  profile: string
}

export interface CanonicalGroupBinding extends CanonicalGroupRoute {
  roomId: string
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
  approval?: { request_id?: string; prompt_id?: string; command?: string; description?: string; choices?: string[]; edit?: unknown }
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

interface GroupSurface {
  mode: GroupExecutionMode
  methods?: string[]
  error?: unknown
}

// Gate mounts share reads within an activation; explicit discovery/submit/retry reads stay fresh.
// Remember only classification, never a route substitute or authority binding.
const groupSurfaces = new Map<string, { epoch?: number; mode?: GroupExecutionMode; read: Promise<GroupSurface> }>()

function advertisedGroupMethods(value: unknown): string[] {
  const methods = (value as { methods?: unknown } | null)?.methods

  return Array.isArray(methods) ? methods.filter((method): method is string => typeof method === 'string') : []
}

export function knownGroupExecutionMode(route: CanonicalGroupRoute): GroupExecutionMode | undefined {
  return groupSurfaces.get(JSON.stringify([route.connectionId, route.profile]))?.mode
}

export function readGroupExecutionMode(route: CanonicalGroupRoute, epoch?: number, refresh = false): Promise<GroupSurface> {
  const key = JSON.stringify([route.connectionId, route.profile])
  const previous = groupSurfaces.get(key)

  if (!refresh && epoch !== undefined && previous?.epoch === epoch) {return previous.read}

  const record = { epoch, mode: previous?.mode, read: canonicalGroupRequest<unknown>(route, 'groups.capabilities')
    .then(value => ({ mode: groupExecutionMode(value), methods: advertisedGroupMethods(value) }))
    .catch(error => ({ mode: groupExecutionMode(undefined, error, previous?.mode), error })) }

  groupSurfaces.set(key, record)

  void record.read.then(result => { record.mode = result.mode })

  return record.read
}

export async function discoverCanonicalGroups(
  route: CanonicalGroupRoute, epoch?: number, refresh = false
): Promise<{ driver: boolean; rooms: CanonicalRoom[] }> {
  const { mode } = await readGroupExecutionMode(route, epoch, refresh)

  if (mode !== 'canonical') {return { driver: false, rooms: [] }}
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

  return { driver: true, rooms }
}

export function canonicalGroupEligibility(
  route: CanonicalGroupRoute, members: GroupMember[]
): { eligible: true; roster: CanonicalRoomMember[] } | { eligible: false; reason: 'classicCount' | 'classicConnection' | 'classicMembers' } {
  if (members.length < 2 || members.length > 6) {return { eligible: false, reason: 'classicCount' }}

  const profiles = new Set<string>()
  const handles = new Set(['all', 'everyone'])
  const roster: CanonicalRoomMember[] = []

  for (const member of members) {
    const connectionId = member.route?.connectionId ?? member.connectionId

    if ((connectionId !== undefined && connectionId !== route.connectionId) ||
      (member.connectionId !== undefined && member.connectionId !== route.connectionId) ||
      (connectionId === undefined && member.remoteSource)) {
      return { eligible: false, reason: 'classicConnection' }
    }

    const profile = member.route?.targetProfile ?? member.targetProfile ?? member.name
    const handle = member.handle ?? profile

    if (!profile.trim() || !handle.trim() || profiles.has(profile.toLowerCase()) || handles.has(handle.toLowerCase())) {
      return { eligible: false, reason: 'classicMembers' }
    }

    profiles.add(profile.toLowerCase())
    handles.add(handle.toLowerCase())
    roster.push({
      member_id: profile, profile, handle,
      target: { kind: 'local', profile },
      ...(member.display_name ? { display_name: member.display_name } : {})
    })
  }

  return { eligible: true, roster }
}

export function isCanonicalGroupCreateRefusal(error: unknown): boolean {
  const refusal = error as { code?: unknown; data?: { reason?: unknown } } | null

  return refusal?.code === 4001 && refusal.data?.reason === 'invalid_params'
}

export async function createCanonicalGroup(
  route: CanonicalGroupRoute,
  name: string,
  members: GroupMember[]
): Promise<{ binding: CanonicalGroupBinding; room: CanonicalRoom }> {
  requireRoute(route)

  if (!name.trim()) {throw new Error('A canonical group needs a name and two to six members')}
  const eligibility = canonicalGroupEligibility(route, members)

  if (!eligibility.eligible) {throw new Error(CANONICAL_GROUP_LOCALES.en[eligibility.reason])}

  const { room } = await canonicalGroupRequest<{ room: CanonicalRoom }>(route, 'groups.create', {
    room_id: crypto.randomUUID(), name, members: eligibility.roster
  })

  return { binding: { ...route, roomId: room.room_id }, room }
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
      throw new Error('Approval requires its exact request ID and an explicit choice')
    }

    if (choice === 'once' && !canonicalApprovalPreview(action).reviewable) {throw new Error('Approval requires a reviewable command or edit')}

    params.request_id = action.request_id
    params.choice = choice
  }

  return canonicalGroupRequest(binding, method, params)
}
