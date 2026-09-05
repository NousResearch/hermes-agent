/**
 * Pure client-side contracts for gateway-hosted Group Chats.
 *
 * This module never talks to a gateway or mutates a room atom. It classifies
 * capability probes, validates the same-gateway boundary, reduces the
 * monotonic event log, and owns the serializable command outbox. Keeping those
 * transitions pure makes reconnect/relaunch behavior testable independently
 * from React and the plugin lifecycle.
 */

import { normalizeHostedMessageIdentity } from './group-message-author'
import type { Attachment, AttachmentKind, GroupMember, GroupMessageAuthor } from './types'

const MIN_ROOM_MEMBERS = 2
const MAX_ROOM_MEMBERS = 6
const MAX_REPLAY_PAGE_SIZE = 500
const MAX_REPLAY_PAGES = 100
const FORBIDDEN_TRANSPORT_FIELD_TOKENS = new Set(['base64', 'byte', 'bytes', 'data', 'path', 'paths'])
const ATTACHMENT_ID_RE = /^att_[0-9a-f]{32}$/
export const ROOM_LINK_PROTOCOL_VERSION = 2

export interface HostedRoomClientLimitations {
  attachmentList?: boolean
  attachments: boolean
  automaticFailover: boolean
  crossGatewayMembers: boolean
}

export const HOSTED_ROOM_CLIENT_LIMITATIONS: HostedRoomClientLimitations = Object.freeze({
  attachmentList: false,
  attachments: false,
  automaticFailover: false,
  crossGatewayMembers: true
})

function hostedCapabilityLimits(capabilities: Record<string, unknown>): HostedRoomClientLimitations {
  const methods = Array.isArray(capabilities.methods) ? capabilities.methods.map(String) : []

  return {
    ...HOSTED_ROOM_CLIENT_LIMITATIONS,
    attachmentList:
      Array.isArray(capabilities.features) && capabilities.features.includes('attachment_metadata_catalog'),
    attachments: methods.includes('groups.attachment.put') && methods.includes('groups.attachment.read')
  }
}

const MAX_HOSTED_ROOM_OUTBOX_COMMANDS = 256

export type HostedRoomCapabilityKind = 'driver-capable' | 'transient-failure' | 'unsupported'

export interface HostedRoomCapability {
  authorityId: null | string
  connectionId: null | string
  exactPeerGrantRevoke: boolean
  kind: HostedRoomCapabilityKind
  limits: typeof HOSTED_ROOM_CLIENT_LIMITATIONS
  maxLogLimit?: number
  persistentProcess: boolean | null
  routeGrantFingerprint: boolean
  reason: null | string
  roomLink: null | RoomLinkCapability
}

export interface RoomLinkCapability {
  catalog: null | {
    attachments: boolean
    digest: null | string
    installationId: null | string
    linkModes: string[]
    persistentProcess: boolean
    protocolVersions: number[]
    text: boolean
  }
  enabled: boolean
  endpoint: null | string
  endpointReason: null | string
  profile: null | string
  reason: null | string
}

export interface AutonomousRoomPlan extends HostedRoomRouteResolution {
  homeConnectionId: null | string
  remoteConnectionIds: string[]
  unavailableConnectionId?: string
}

export interface HostedRoomRouteResolution {
  connectionId: null | string
  kind: 'multi-gateway' | 'single-gateway' | 'unsupported'
  limits: typeof HOSTED_ROOM_CLIENT_LIMITATIONS
  memberConnectionIds: Array<null | string>
  reason: null | string
}

export interface HostedRoomEvent {
  actor: Record<string, unknown>
  createdAt: number
  eventId: string
  kind: string
  payload: Record<string, unknown>
  roomId: null | string
  seq: number
}

export interface HostedReplayMessage {
  at: number
  eventId: string
  roomId?: string
  from: GroupMessageAuthor
  images?: Attachment[]
  seq: number
  text: string
  thread: string
}

function replayAttachments(value: unknown): Attachment[] {
  if (!Array.isArray(value) || value.length > 8) {
    return []
  }

  const result: Attachment[] = []

  for (const raw of value) {
    const item = record(raw)
    const attachmentId = text(item?.attachment_id)
    const kind = text(item?.kind) as AttachmentKind | null
    const name = text(item?.name)
    const mime = text(item?.mime)
    const size = Number(item?.size)

    if (
      !attachmentId ||
      !ATTACHMENT_ID_RE.test(attachmentId) ||
      !kind ||
      !['image', 'pdf', 'file'].includes(kind) ||
      !name ||
      !mime ||
      !Number.isSafeInteger(size) ||
      size < 0 ||
      size > 15_000_000
    ) {
      continue
    }

    result.push({ attachmentId, kind, mime, name, size })
  }

  return result
}

export interface HostedRoomActivity {
  at: number
  eventId: string
  kind: string
  member: string
  reasonCode: null | string
  seq: number
}

export interface HostedRoomReplayState {
  activity: HostedRoomActivity[]
  authorityEpoch: null | number
  authorityId: null | string
  conflicts: Array<{ eventId: string; seq: number }>
  connectionId: null | string
  cursor: number
  deleted: boolean
  lastStatusEvent: HostedRoomEvent | null
  latestSeq: number
  members: Array<Record<string, unknown>>
  messages: HostedReplayMessage[]
  name: string
  pendingEvents: HostedRoomEvent[]
  roomId: null | string
  timeline: Array<{ eventId: string; kind: string; seq: number }>
}

export interface FriendlyHostedRoomStatus {
  canRetry?: boolean
  canStop?: boolean
  kind: string
  member?: null | string
  reasonCode?: null | string
}

export type HostedRoomCommandKind = 'create' | 'disband' | 'rename' | 'retry' | 'send' | 'stop'
export type HostedRoomCommandStatus = 'failed' | 'in-flight' | 'pending'

export interface HostedRoomCommand {
  attempts: number
  authorityId: null | string
  commandId: string
  connectionId: string
  failureCode: null | string
  kind: HostedRoomCommandKind
  payload: Record<string, unknown>
  roomId: string
  status: HostedRoomCommandStatus
}

export interface HostedRoomOutbox {
  commands: HostedRoomCommand[]
  version: 1
}

export type HostedRoomOutboxAction =
  | { command: Partial<HostedRoomCommand>; type: 'enqueue' }
  | { command: Partial<HostedRoomCommand>; type: 'enqueue-safety' }
  | { commandId: string; type: 'acknowledge' | 'dispatch' | 'retry' | 'transient-failure' }
  | { commandId: string; failureCode?: string; type: 'terminal-failure' }

const STATUS_EVENT_KINDS = new Set([
  'authority.lost',
  'member.unavailable',
  'room.activity',
  'turn.cancelled',
  'turn.deferred',
  'turn.failed',
  'turn.reassigned',
  'turn.settled',
  'turn.started'
])

const KNOWN_EVENT_KINDS = new Set([
  'authority.claimed',
  'authority.lost',
  'member.unavailable',
  'message.member',
  'message.user',
  'room.activity',
  'room.created',
  'room.disbanded',
  'room.members_changed',
  'room.renamed',
  'turn.cancelled',
  'turn.deferred',
  'turn.failed',
  'turn.reassigned',
  'turn.settled',
  'turn.started'
])

function record(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>) : null
}

function text(value: unknown): null | string {
  return typeof value === 'string' && value.trim() ? value.trim() : null
}

function nonNegativeInteger(value: unknown, fallback = 0): number {
  const number = Number(value)

  return Number.isSafeInteger(number) && number >= 0 ? number : fallback
}

function positiveInteger(value: unknown, fallback: null | number = null): null | number {
  const number = Number(value)

  return Number.isSafeInteger(number) && number > 0 ? number : fallback
}

function timestampMilliseconds(value: unknown) {
  const number = Number(value)

  if (!Number.isFinite(number) || number <= 0) {
    return 0
  }

  return number < 1_000_000_000_000 ? number * 1000 : number
}

function errorCode(error: unknown): unknown {
  const outer = record(error)
  const inner = record(outer?.error)

  return outer?.code ?? inner?.code ?? null
}

function errorMessage(error: unknown): string {
  const outer = record(error)
  const inner = record(outer?.error)

  return String(outer?.message || inner?.message || error || '')
}

function isMissingCapabilityMethod(error: unknown): boolean {
  return (
    errorCode(error) === -32601 ||
    /method not found|-32601|unknown method|no such method|no handler for|unsupported rpc/i.test(errorMessage(error))
  )
}

function capabilityResult(probe: unknown): Record<string, unknown> | null {
  const candidate = record(probe)

  if (!candidate) {
    return null
  }

  if (candidate.ok === true) {
    return record(candidate.result)
  }

  if (
    !Object.prototype.hasOwnProperty.call(candidate, 'ok') &&
    !Object.prototype.hasOwnProperty.call(candidate, 'error')
  ) {
    return candidate
  }

  return null
}

function roomLinkCapability(value: unknown): null | RoomLinkCapability {
  const candidate = record(value)

  if (!candidate) {
    return null
  }

  const catalog = record(candidate.catalog)
  const endpoint = record(candidate.endpoint)

  return {
    enabled: candidate.enabled === true,
    endpoint: endpoint?.available === true ? text(endpoint.url) : null,
    endpointReason: endpoint?.available === false ? text(endpoint.reason) : null,
    reason: text(candidate.reason),
    profile: text(candidate.profile),
    catalog: catalog
      ? {
          installationId: text(catalog.installation_id),
          digest: text(catalog.catalog_digest),
          persistentProcess: catalog.persistent_process === true,
          text: catalog.text === true,
          attachments: catalog.attachments === true,
          linkModes: Array.isArray(catalog.link_modes) ? catalog.link_modes.map(String).filter(Boolean) : [],
          protocolVersions: Array.isArray(catalog.protocol_versions)
            ? catalog.protocol_versions.map(Number).filter(Number.isSafeInteger)
            : []
        }
      : null
  }
}

/** A missing RPC is a compatibility verdict; a socket failure is not. */
export function classifyHostedRoomCapability(
  probe: unknown,
  { connectionId = null }: { connectionId?: null | string } = {}
): HostedRoomCapability {
  const candidate = record(probe)
  const error = probe instanceof Error ? probe : candidate?.ok === false ? candidate.error || probe : candidate?.error
  const localConnectionId = text(connectionId)

  if (error) {
    const unsupported = isMissingCapabilityMethod(error)

    return {
      kind: unsupported ? 'unsupported' : 'transient-failure',
      reason: unsupported ? 'old-gateway' : 'probe-failed',
      connectionId: localConnectionId,
      exactPeerGrantRevoke: false,
      authorityId: null,
      persistentProcess: null,
      routeGrantFingerprint: false,
      roomLink: null,
      limits: HOSTED_ROOM_CLIENT_LIMITATIONS
    }
  }

  const capabilities = capabilityResult(probe)

  if (!capabilities) {
    return {
      kind: 'transient-failure',
      reason: 'invalid-response',
      connectionId: localConnectionId,
      exactPeerGrantRevoke: false,
      authorityId: null,
      persistentProcess: null,
      routeGrantFingerprint: false,
      roomLink: null,
      limits: HOSTED_ROOM_CLIENT_LIMITATIONS
    }
  }

  if (capabilities.driver !== true) {
    return {
      kind: 'unsupported',
      reason: capabilities.driver === false ? 'driver-disabled' : 'incomplete-contract',
      connectionId: localConnectionId,
      exactPeerGrantRevoke: false,
      authorityId: capabilities.driver === false ? text(capabilities.authority_gateway_id) : null,
      persistentProcess: capabilities.persistent_process === true,
      routeGrantFingerprint: false,
      roomLink: roomLinkCapability(capabilities.room_link),
      limits: {
        ...HOSTED_ROOM_CLIENT_LIMITATIONS,
        attachmentList: capabilities.driver === false && hostedCapabilityLimits(capabilities).attachmentList
      }
    }
  }

  const authorityId = text(capabilities.authority_gateway_id)

  if (!authorityId) {
    return {
      kind: 'unsupported',
      reason: 'incomplete-contract',
      connectionId: localConnectionId,
      exactPeerGrantRevoke: false,
      authorityId: null,
      persistentProcess: capabilities.persistent_process === true,
      routeGrantFingerprint: false,
      roomLink: roomLinkCapability(capabilities.room_link),
      limits: HOSTED_ROOM_CLIENT_LIMITATIONS
    }
  }

  return {
    kind: 'driver-capable',
    reason: null,
    connectionId: localConnectionId,
    exactPeerGrantRevoke:
      Array.isArray(capabilities.methods) && capabilities.methods.includes('groups.peer.revoke_exact'),
    authorityId,
    persistentProcess: capabilities.persistent_process === true,
    routeGrantFingerprint:
      Array.isArray(capabilities.features) && capabilities.features.includes('peer_route_grant_fingerprint'),
    roomLink: roomLinkCapability(capabilities.room_link),
    maxLogLimit: positiveInteger(capabilities.max_log_limit, 100) || 100,
    limits: hostedCapabilityLimits(capabilities)
  }
}

/** Stored room reads do not require a live, persistent execution worker. */
export function isHostedRoomReadEligible(capability: HostedRoomCapability): boolean {
  return (
    Boolean(capability.authorityId) && (capability.kind === 'driver-capable' || capability.reason === 'driver-disabled')
  )
}

export function isHostedRoomContinuityEligible(capability: unknown): boolean {
  const candidate = record(capability)

  if (!candidate) {
    return false
  }

  if (Object.prototype.hasOwnProperty.call(candidate, 'kind')) {
    return candidate.kind === 'driver-capable' && candidate.persistentProcess === true
  }

  return candidate.driver === true && candidate.persistent_process === true
}

function memberConnectionId(member: GroupMember, activeConnectionId: null | string): null | string {
  if (!member || member.sourceMissing) {
    return null
  }

  const explicit = text(member.route?.connectionId) || text(member.connectionId)

  if (explicit) {
    return explicit
  }

  if (member.sourceScoped || member.remoteSource) {
    return null
  }

  return text(activeConnectionId)
}

export function resolveSingleGatewayRoute(
  members: GroupMember[],
  { activeConnectionId = null }: { activeConnectionId?: null | string } = {}
): HostedRoomRouteResolution {
  const roster = Array.isArray(members) ? members : []

  if (roster.length < MIN_ROOM_MEMBERS || roster.length > MAX_ROOM_MEMBERS) {
    return {
      kind: 'unsupported',
      reason: 'member-count',
      connectionId: null,
      memberConnectionIds: [],
      limits: HOSTED_ROOM_CLIENT_LIMITATIONS
    }
  }

  const memberConnectionIds = roster.map(member => memberConnectionId(member, activeConnectionId))

  if (memberConnectionIds.some(connectionId => !connectionId)) {
    return {
      kind: 'unsupported',
      reason: 'unresolved-member-route',
      connectionId: null,
      memberConnectionIds,
      limits: HOSTED_ROOM_CLIENT_LIMITATIONS
    }
  }

  const distinct = new Set(memberConnectionIds)

  if (distinct.size !== 1) {
    return {
      kind: 'unsupported',
      reason: 'cross-gateway',
      connectionId: null,
      memberConnectionIds,
      limits: HOSTED_ROOM_CLIENT_LIMITATIONS
    }
  }

  return {
    kind: 'single-gateway',
    reason: null,
    connectionId: memberConnectionIds[0],
    memberConnectionIds,
    limits: HOSTED_ROOM_CLIENT_LIMITATIONS
  }
}

/** Choose the simplest autonomous plan without widening any gateway's
 * advertised capability. */
export function resolveAutonomousRoomPlan(
  members: GroupMember[],
  {
    activeConnectionId = null,
    capabilities = {}
  }: {
    activeConnectionId?: null | string
    capabilities?: Record<string, HostedRoomCapability | null>
  } = {}
): AutonomousRoomPlan {
  const roster = Array.isArray(members) ? members : []

  const route = resolveSingleGatewayRoute(roster, {
    activeConnectionId
  })

  if (route.reason && route.reason !== 'cross-gateway') {
    return {
      ...route,
      homeConnectionId: null,
      remoteConnectionIds: []
    }
  }

  const memberConnectionIds = roster.map(member => memberConnectionId(member, activeConnectionId))
  const connectionIds = [...new Set(memberConnectionIds.filter((value): value is string => Boolean(value)))]

  const homeCandidates = connectionIds.filter(connectionId => {
    const capability = capabilities[connectionId]

    if (capability?.kind !== 'driver-capable' || capability.persistentProcess !== true) {
      return false
    }

    if (connectionIds.length === 1) {
      return true
    }

    const roomLink = capability.roomLink

    return Boolean(
      roomLink?.enabled === true &&
      roomLink.catalog?.persistentProcess === true &&
      roomLink.catalog.protocolVersions.includes(ROOM_LINK_PROTOCOL_VERSION) &&
      roomLink.catalog.linkModes.includes('direct')
    )
  })

  const preferredHome = text(activeConnectionId)

  const homeConnectionId =
    preferredHome && homeCandidates.includes(preferredHome) ? preferredHome : homeCandidates[0] || null

  if (!homeConnectionId) {
    return {
      kind: 'unsupported',
      reason: 'no-persistent-home',
      connectionId: null,
      homeConnectionId: null,
      memberConnectionIds,
      remoteConnectionIds: connectionIds,
      limits: HOSTED_ROOM_CLIENT_LIMITATIONS
    }
  }

  const remoteConnectionIds = connectionIds.filter(connectionId => connectionId !== homeConnectionId)

  const unsupportedRemote = remoteConnectionIds.find(connectionId => {
    const roomLink = capabilities[connectionId]?.roomLink

    return !(
      roomLink?.enabled === true &&
      roomLink.catalog?.persistentProcess === true &&
      roomLink.catalog.text === true &&
      roomLink.catalog.installationId &&
      roomLink.catalog.digest &&
      roomLink.catalog.protocolVersions.includes(ROOM_LINK_PROTOCOL_VERSION) &&
      roomLink.catalog.linkModes.includes('direct')
    )
  })

  if (unsupportedRemote) {
    return {
      kind: 'unsupported',
      reason: 'remote-needs-setup',
      connectionId: null,
      homeConnectionId,
      memberConnectionIds,
      remoteConnectionIds,
      unavailableConnectionId: unsupportedRemote,
      limits: HOSTED_ROOM_CLIENT_LIMITATIONS
    }
  }

  const unreachableRemote = remoteConnectionIds.find(connectionId => !capabilities[connectionId]?.roomLink?.endpoint)

  if (unreachableRemote) {
    return {
      kind: 'unsupported',
      reason: 'remote-needs-address',
      connectionId: null,
      homeConnectionId,
      memberConnectionIds,
      remoteConnectionIds,
      unavailableConnectionId: unreachableRemote,
      limits: HOSTED_ROOM_CLIENT_LIMITATIONS
    }
  }

  return {
    kind: remoteConnectionIds.length ? 'multi-gateway' : 'single-gateway',
    reason: null,
    connectionId: homeConnectionId,
    homeConnectionId,
    memberConnectionIds,
    remoteConnectionIds,
    limits: HOSTED_ROOM_CLIENT_LIMITATIONS
  }
}

export function describeAutonomousRoomPlan(
  plan: AutonomousRoomPlan,
  { homeLabel = 'one host', unavailableLabel = 'One host' } = {}
) {
  if (plan.kind === 'multi-gateway') {
    return {
      defaultEnabled: true,
      level: 'distributed' as const,
      title: 'Works without Desktop',
      description: 'Bots can continue while Desktop is closed.'
    }
  }

  if (plan.kind === 'single-gateway') {
    return {
      defaultEnabled: true,
      level: 'gateway' as const,
      title: 'Works without Desktop',
      description: 'Bots can continue while Desktop is closed.'
    }
  }

  const needsSetup = ['remote-needs-address', 'remote-needs-setup'].includes(String(plan.reason || ''))

  return {
    defaultEnabled: false,
    level: 'desktop' as const,
    title: 'Keep Desktop open',
    description: needsSetup
      ? `${unavailableLabel} can't keep this Group Chat running yet.`
      : 'Bots pause when Desktop closes.'
  }
}

export function profileScopedRoomLinkEndpoint(endpoint: unknown, profile: unknown) {
  const base = text(endpoint)?.replace(/\/+$/, '') || null
  const targetProfile = text(profile)

  if (!base || !targetProfile || targetProfile === 'default') {
    return base
  }

  const suffix = `/p/${encodeURIComponent(targetProfile)}`

  if (base.endsWith(suffix)) {
    return base
  }

  if (/\/p\/[^/]+$/i.test(base)) {
    return null
  }

  return `${base}${suffix}`
}

export function describeHostedRoomCreationError(error: unknown) {
  const message = errorMessage(error)

  if (/unreachable|name or service not known|timed? ?out|connection refused|network is unreachable/i.test(message)) {
    return 'One Bot host cannot reach another. Check that both are online, then try again.'
  }

  if (/non-json|authorization|grant|renewal|http 40[13]|unknown or unconfigured profile/i.test(message)) {
    return 'One Bot host could not verify this Group Chat. Update or reconnect it, then try again.'
  }

  if (/capability catalog changed|changed during setup/i.test(message)) {
    return 'A Bot host changed while the Group Chat was being created. Wait for it to reconnect, then try again.'
  }

  return null
}

function cloneRecords(value: unknown): Array<Record<string, unknown>> {
  return Array.isArray(value) ? value.map(item => ({ ...(record(item) || {}) })) : []
}

export function createHostedRoomReplayState(
  input: Partial<HostedRoomReplayState> & { roomId?: null | string } = {}
): HostedRoomReplayState {
  const cursor = nonNegativeInteger(input.cursor)

  return {
    roomId: text(input.roomId),
    name: typeof input.name === 'string' ? input.name : '',
    members: cloneRecords(input.members),
    authorityId: text(input.authorityId),
    authorityEpoch: positiveInteger(input.authorityEpoch),
    connectionId: text(input.connectionId),
    cursor,
    latestSeq: Math.max(cursor, nonNegativeInteger(input.latestSeq, cursor)),
    messages: Array.isArray(input.messages) ? input.messages.map(message => ({ ...message })) : [],
    activity: Array.isArray(input.activity) ? input.activity.map(entry => ({ ...entry })) : [],
    timeline: Array.isArray(input.timeline) ? input.timeline.map(entry => ({ ...entry })) : [],
    pendingEvents: Array.isArray(input.pendingEvents) ? input.pendingEvents.map(entry => ({ ...entry })) : [],
    lastStatusEvent: input.lastStatusEvent ? { ...input.lastStatusEvent } : null,
    deleted: Boolean(input.deleted),
    conflicts: Array.isArray(input.conflicts) ? input.conflicts.map(conflict => ({ ...conflict })) : []
  }
}

function normalizeEvent(raw: unknown): HostedRoomEvent | null {
  const candidate = record(raw)

  if (!candidate) {
    return null
  }

  const seq = positiveInteger(candidate.seq)
  const eventId = text(candidate.event_id) || text(candidate.eventId)
  const kind = text(candidate.kind)

  if (!seq || !eventId || !kind) {
    return null
  }

  return {
    roomId: text(candidate.room_id) || text(candidate.roomId),
    seq,
    eventId,
    kind,
    actor: { ...(record(candidate.actor) || {}) },
    payload: { ...(record(candidate.payload) || {}) },
    createdAt: timestampMilliseconds(candidate.created_at ?? candidate.createdAt)
  }
}

function memberLabel(event: HostedRoomEvent): string {
  return (
    text(event.payload.member_display_name) ||
    text(event.payload.member_name) ||
    text(event.payload.display_name) ||
    text(event.actor.display_name) ||
    text(event.actor.profile) ||
    ''
  )
}

function messageFromEvent(event: HostedRoomEvent, roomId: null | string): HostedReplayMessage {
  const user = event.kind === 'message.user'

  const identity =
    !user && event.actor.kind === 'member'
      ? normalizeHostedMessageIdentity({
          roomId: event.roomId || roomId,
          memberId: event.actor.id,
          profile: event.actor.profile
        })
      : undefined

  const images = replayAttachments(event.payload.attachments)

  return {
    seq: event.seq,
    eventId: event.eventId,
    ...(user && (event.roomId || roomId) ? { roomId: (event.roomId || roomId)! } : {}),
    from: {
      kind: user ? 'user' : 'member',
      name: user ? 'You' : memberLabel(event),
      ...(identity ? { hostedIdentity: identity } : {}),
      ...(text(event.actor.connection_id) ? { source: text(event.actor.connection_id) || undefined } : {})
    },
    text: typeof event.payload.text === 'string' ? event.payload.text : '',
    thread: text(event.payload.thread_id) || text(event.payload.thread) || 'legacy',
    at: event.createdAt,
    ...(images.length ? { images } : {})
  }
}

function applyReplayEvent(state: HostedRoomReplayState, event: HostedRoomEvent): void {
  if (KNOWN_EVENT_KINDS.has(event.kind)) {
    state.timeline.push({
      seq: event.seq,
      eventId: event.eventId,
      kind: event.kind
    })
  }

  if (event.kind === 'message.user' || event.kind === 'message.member') {
    state.messages.push(messageFromEvent(event, state.roomId))
  } else if (event.kind === 'room.created') {
    state.name = text(event.payload.name) || state.name

    if (Array.isArray(event.payload.members)) {
      state.members = cloneRecords(event.payload.members)
    }
  } else if (event.kind === 'room.renamed') {
    state.name = text(event.payload.name) || state.name
  } else if (event.kind === 'room.members_changed' && Array.isArray(event.payload.members)) {
    state.members = cloneRecords(event.payload.members)
  } else if (event.kind === 'room.disbanded') {
    state.deleted = true
  } else if (event.kind === 'authority.claimed') {
    const authorityId = text(event.payload.authority_gateway_id)

    if (authorityId) {
      if (state.authorityId && state.authorityId !== authorityId) {
        state.connectionId = null
      }

      state.authorityId = authorityId
    }

    state.authorityEpoch = positiveInteger(event.payload.authority_epoch, state.authorityEpoch)
  } else if (event.kind === 'authority.lost') {
    state.connectionId = null
  }

  if (STATUS_EVENT_KINDS.has(event.kind)) {
    state.activity.push({
      seq: event.seq,
      eventId: event.eventId,
      kind: event.kind,
      member: memberLabel(event),
      reasonCode: text(event.payload.reason_code),
      at: event.createdAt
    })
    state.lastStatusEvent = event
  }
}

export function reduceHostedRoomEvents(state: HostedRoomReplayState, incomingEvents: unknown[]): HostedRoomReplayState {
  const next = createHostedRoomReplayState(state)
  const bySeq = new Map<number, HostedRoomEvent>()
  const byId = new Map<string, HostedRoomEvent>()

  for (const candidate of [...next.pendingEvents, ...(Array.isArray(incomingEvents) ? incomingEvents : [])]) {
    const event = normalizeEvent(candidate)

    if (!event || event.seq <= next.cursor || (next.roomId && event.roomId && next.roomId !== event.roomId)) {
      continue
    }

    const prior = bySeq.get(event.seq) || byId.get(event.eventId)

    if (prior) {
      if (prior.seq !== event.seq || prior.eventId !== event.eventId) {
        next.conflicts.push({
          seq: event.seq,
          eventId: event.eventId
        })
      }

      continue
    }

    bySeq.set(event.seq, event)
    byId.set(event.eventId, event)
  }

  const pending = [...bySeq.values()].sort(
    (left, right) => left.seq - right.seq || left.eventId.localeCompare(right.eventId)
  )

  next.latestSeq = Math.max(next.latestSeq, ...pending.map(event => event.seq), next.cursor)

  while (pending.length && pending[0].seq === next.cursor + 1) {
    const event = pending.shift()

    if (!event) {
      break
    }

    applyReplayEvent(next, event)
    next.cursor = event.seq
  }

  next.pendingEvents = pending

  return next
}

function status(
  kind: string,
  options: Pick<FriendlyHostedRoomStatus, 'canRetry' | 'canStop' | 'member' | 'reasonCode'> = {}
): FriendlyHostedRoomStatus {
  return {
    kind,
    ...options
  }
}

export function deriveFriendlyHostedRoomStatus(state: HostedRoomReplayState): FriendlyHostedRoomStatus {
  if (state.deleted) {
    return status('deleted')
  }

  if (state.authorityId && !state.connectionId) {
    return status('offline', {
      canRetry: true
    })
  }

  const event = state.lastStatusEvent

  if (!event) {
    return status('ready')
  }

  const member = memberLabel(event)

  if (event.kind === 'turn.started' || event.kind === 'turn.reassigned') {
    return status('working', {
      member,
      canStop: true
    })
  }

  if (event.kind === 'member.unavailable') {
    return status('member-unavailable', {
      member,
      canRetry: true
    })
  }

  if (event.kind === 'turn.failed') {
    const reason = text(event.payload.reason_code)

    const needsAttention = [
      'provider_auth_or_access',
      'provider_quota_limit',
      'missing_config',
      'agent_blocked'
    ].includes(reason || '')

    return status(needsAttention ? 'needs-attention' : 'failed', {
      member,
      reasonCode: reason,
      canRetry: !needsAttention
    })
  }

  if (event.kind === 'turn.deferred') {
    return status('waiting', {
      member,
      canRetry: true
    })
  }

  if (event.kind === 'turn.cancelled') {
    return status('stopped')
  }

  if (event.kind === 'turn.settled') {
    return status('ready')
  }

  if (event.kind === 'room.activity') {
    const activity = text(event.payload.status)?.toLowerCase()

    if (activity === 'working') {
      return status('working', {
        member,
        canStop: true
      })
    }

    if (activity === 'needs_user' || activity === 'waiting_for_user') {
      return status('needs-you')
    }
  }

  return status('ready')
}

export async function replayHostedRoomPages({
  state,
  fetchPage,
  pageSize = 100,
  maxPages = 20
}: {
  fetchPage: (request: { limit: number; sinceSeq: number }) => Promise<unknown>
  maxPages?: number
  pageSize?: number
  state: HostedRoomReplayState
}) {
  const limit = Math.min(MAX_REPLAY_PAGE_SIZE, Math.max(1, positiveInteger(pageSize, 100) || 100))
  const pageBound = Math.min(MAX_REPLAY_PAGES, Math.max(1, positiveInteger(maxPages, 20) || 20))
  let next = createHostedRoomReplayState(state)
  let pages = 0
  let fetchedEvents = 0

  while (pages < pageBound) {
    const beforeCursor = next.cursor
    let rawPage: unknown

    try {
      rawPage = await fetchPage({
        sinceSeq: beforeCursor,
        limit
      })
    } catch (error) {
      return {
        state: next,
        complete: false,
        reason: 'transient-failure',
        pages,
        fetchedEvents,
        error
      }
    }

    const page = record(rawPage)

    if (!page) {
      return {
        state: next,
        complete: false,
        reason: 'invalid-response',
        pages,
        fetchedEvents
      }
    }

    const events = Array.isArray(page.events) ? page.events : []
    const latestSeq = nonNegativeInteger(page.latest_seq ?? page.latestSeq, next.latestSeq)

    if (events.length > limit) {
      return {
        state: next,
        complete: false,
        reason: 'oversized-page',
        pages,
        fetchedEvents
      }
    }

    pages += 1
    fetchedEvents += events.length
    next = reduceHostedRoomEvents(next, events)
    next.latestSeq = Math.max(next.latestSeq, latestSeq)

    const hasMore = page.has_more === true || next.cursor < latestSeq

    if (!hasMore) {
      return {
        state: next,
        complete: next.pendingEvents.length === 0,
        reason: next.pendingEvents.length ? 'gap' : null,
        pages,
        fetchedEvents
      }
    }

    if (next.cursor <= beforeCursor) {
      return {
        state: next,
        complete: false,
        reason: 'stalled',
        pages,
        fetchedEvents
      }
    }
  }

  return {
    state: next,
    complete: false,
    reason: 'limit',
    pages,
    fetchedEvents
  }
}

function jsonRecord(value: unknown, label: string): Record<string, unknown> {
  let cloned: unknown

  try {
    cloned = JSON.parse(JSON.stringify(value ?? {})) as unknown
  } catch (error) {
    throw new TypeError(`${label} must be JSON-serializable`, {
      cause: error
    })
  }

  const result = record(cloned) || {}

  assertNoRawTransportFields(result)

  return result
}

function fieldTokens(field: string) {
  return String(field)
    .replace(/([a-z0-9])([A-Z])/g, '$1_$2')
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter(Boolean)
}

function assertNoRawTransportFields(value: unknown, location = 'payload'): void {
  if (!value || typeof value !== 'object') {
    return
  }

  if (Array.isArray(value)) {
    value.forEach((entry, index) => assertNoRawTransportFields(entry, `${location}[${index}]`))

    return
  }

  for (const [field, nested] of Object.entries(value)) {
    const forbidden = fieldTokens(field).find(token => FORBIDDEN_TRANSPORT_FIELD_TOKENS.has(token))

    if (forbidden) {
      throw new TypeError(`${location}.${field} cannot carry raw attachment ${forbidden}`)
    }

    assertNoRawTransportFields(nested, `${location}.${field}`)
  }
}

function stableJson(value: unknown): string {
  if (Array.isArray(value)) {
    return `[${value.map(stableJson).join(',')}]`
  }

  if (value && typeof value === 'object') {
    return `{${Object.entries(value)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, nested]) => `${JSON.stringify(key)}:${stableJson(nested)}`)
      .join(',')}}`
  }

  return JSON.stringify(value)
}

function normalizeCommand(raw: Partial<HostedRoomCommand>): HostedRoomCommand {
  const commandId = text(raw.commandId)
  const kind = text(raw.kind)
  const roomId = text(raw.roomId)
  const connectionId = text(raw.connectionId)

  if (
    !commandId ||
    !kind ||
    !['create', 'disband', 'rename', 'retry', 'send', 'stop'].includes(kind) ||
    !roomId ||
    !connectionId
  ) {
    throw new TypeError('hosted room command is incomplete')
  }

  return {
    commandId,
    kind: kind as HostedRoomCommandKind,
    roomId,
    authorityId: text(raw.authorityId),
    connectionId,
    payload: jsonRecord(raw.payload, 'command payload'),
    status: ['failed', 'in-flight', 'pending'].includes(String(raw.status))
      ? (raw.status as HostedRoomCommandStatus)
      : 'pending',
    attempts: nonNegativeInteger(raw.attempts),
    failureCode: text(raw.failureCode)
  }
}

function commandSignature(command: HostedRoomCommand): string {
  return stableJson({
    commandId: command.commandId,
    kind: command.kind,
    roomId: command.roomId,
    authorityId: command.authorityId,
    connectionId: command.connectionId,
    payload: command.payload
  })
}

export function createHostedRoomOutbox(
  persisted: unknown = null,
  recoverInFlight = true,
  tolerateInvalid = false
): HostedRoomOutbox {
  const candidate = record(persisted)
  const commands: HostedRoomCommand[] = []
  const quarantinedRooms = new Set<string>()
  let quarantineTail = false

  for (const raw of Array.isArray(candidate?.commands) ? candidate.commands : []) {
    const rawRoomId = text(record(raw)?.roomId)

    if (quarantineTail || (rawRoomId && quarantinedRooms.has(rawRoomId))) {
      continue
    }

    try {
      const command = normalizeCommand((record(raw) || {}) as Partial<HostedRoomCommand>)
      const existing = commands.find(entry => entry.commandId === command.commandId)

      command.status = recoverInFlight && command.status === 'in-flight' ? 'pending' : command.status

      if (!existing) {
        commands.push(command)
      } else if (commandSignature(existing) !== commandSignature(command)) {
        throw new TypeError(`commandId ${command.commandId} has conflicting persisted content`)
      }
    } catch (error) {
      if (!tolerateInvalid) {
        throw error
      }

      if (rawRoomId) {
        quarantinedRooms.add(rawRoomId)
      } else {
        // Without a room identity, no later row can be proven independent of
        // the damaged command. Keep only the known-good prefix.
        quarantineTail = true
      }
    }
  }

  return {
    version: 1,
    commands
  }
}

export function reduceHostedRoomOutbox(state: HostedRoomOutbox, action: HostedRoomOutboxAction): HostedRoomOutbox {
  const current = state && Array.isArray(state.commands) ? state : createHostedRoomOutbox()

  if (action.type === 'enqueue' || action.type === 'enqueue-safety') {
    const command = normalizeCommand(action.command)
    const roomHasFailure = current.commands.some(entry => entry.roomId === command.roomId && entry.status === 'failed')

    const commands =
      action.type === 'enqueue-safety' && roomHasFailure
        ? current.commands.filter(entry => entry.roomId !== command.roomId)
        : current.commands

    const existing = commands.find(entry => entry.commandId === command.commandId)

    command.status = 'pending'

    if (existing) {
      if (commandSignature(existing) !== commandSignature(command)) {
        throw new TypeError(`commandId ${command.commandId} is already bound to different content`)
      }

      return current
    }

    if (commands.filter(entry => entry.status !== 'failed').length >= MAX_HOSTED_ROOM_OUTBOX_COMMANDS) {
      throw new TypeError(
        'Too many Group Chat changes are waiting to sync. Reconnect the affected device and try again.'
      )
    }

    return {
      ...current,
      commands: [...commands, command]
    }
  }

  const commandId = text(action.commandId)

  if (!commandId) {
    throw new TypeError('outbox action requires commandId')
  }

  if (action.type === 'acknowledge') {
    return {
      ...current,
      commands: current.commands.filter(command => command.commandId !== commandId)
    }
  }

  return {
    ...current,
    commands: current.commands.map(command => {
      if (command.commandId !== commandId) {
        return command
      }

      if (action.type === 'dispatch') {
        return {
          ...command,
          status: 'in-flight' as const,
          attempts: command.attempts + 1,
          failureCode: null
        }
      }

      if (action.type === 'terminal-failure') {
        return {
          ...command,
          status: 'failed' as const,
          failureCode: text(action.failureCode) || 'command-failed'
        }
      }

      if (action.type === 'retry') {
        return {
          ...command,
          status: 'pending' as const,
          attempts: 0,
          failureCode: null
        }
      }

      return {
        ...command,
        status: 'pending' as const
      }
    })
  }
}
