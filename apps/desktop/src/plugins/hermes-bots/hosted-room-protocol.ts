import type { GroupChat, GroupMember, GroupMessage } from './types'

/** Never use a display name (or a bare room id) to route hosted work. */
export interface HostedRoomIdentity {
  connectionId: string
  authorityGatewayId: string
  roomId: string
}

export interface HostedRoomSummary {
  room_id: string
  name: string
  authority_gateway_id: string
  authority_epoch: number
  members: HostedMember[]
  latest_seq: number
}

interface HostedMember {
  member_id: string
  profile: string
  handle: string
  display_name?: string
}

export interface HostedEvent {
  room_id: string
  seq: number
  event_id: string
  kind: string
  actor: { kind: string; id: string; profile?: string }
  payload: Record<string, unknown>
  created_at: number
  authority_epoch: number | null
}

export interface HostedCapabilities {
  authorityGatewayId: string
  logLimit: number
  driver: boolean
  persistentProcess: boolean
  persistentHolds: boolean
}

/** One member the user has paused, as the owning gateway reports it. */
export interface HostedHold {
  memberId: string
  handle: string
  label: string
}

export interface HostedReplay {
  cursor: number
  events: HostedEvent[]
}

export function hostedRoomKey(identity: HostedRoomIdentity): string {
  return `hosted:${JSON.stringify([identity.connectionId, identity.authorityGatewayId, identity.roomId])}`
}

export function isHostedRoomKey(key: string): boolean {
  return key.startsWith('hosted:[')
}

export function hostedRecord(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error('Invalid hosted room response')
  }

  return value as Record<string, unknown>
}

function text(value: unknown): string {
  if (typeof value !== 'string' || !value) {throw new Error('Invalid hosted room identity or text')}

  return value
}

function integer(value: unknown, minimum = 0): number {
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < minimum) {
    throw new Error('Invalid hosted room sequence')
  }

  return value
}

export function parseHostedCapabilities(raw: unknown): HostedCapabilities {
  const value = hostedRecord(raw)
  const methods = value.methods
  const features = value.features

  if (value.protocol_version !== 2 || !Array.isArray(methods) || !Array.isArray(features) ||
      !['groups.list', 'groups.state', 'groups.send', 'groups.log', 'groups.stop'].every(m => methods.includes(m)) ||
      !['room_identity', 'monotonic_log', 'idempotent_send', 'typed_events'].every(f => features.includes(f))) {
    throw new Error('This gateway does not support hosted Group Chat protocol v2')
  }

  return {
    authorityGatewayId: text(value.authority_gateway_id),
    logLimit: Math.min(100, integer(value.max_log_limit, 1)),
    driver: value.driver === true,
    persistentProcess: value.persistent_process === true,
    // Optional on purpose: a gateway without durable holds still serves this room, it just
    // cannot promise a pause survives. Requiring the feature would push it into a fallback.
    persistentHolds: features.includes('persistent_member_holds')
  }
}

/** Members the backend reports as held. The hold itself is backend-owned state: this only
 * labels it, and never infers a hold from local text or from a member's silence. */
export function hostedHolds(driverStatus: unknown): HostedHold[] {
  const status = driverStatus && typeof driverStatus === 'object' ? driverStatus as Record<string, unknown> : {}

  if (!Array.isArray(status.holds)) {return []}

  return status.holds.flatMap(raw => {
    if (!raw || typeof raw !== 'object') {return []}
    const hold = raw as Record<string, unknown>
    const memberId = typeof hold.member_id === 'string' ? hold.member_id : ''

    if (!memberId) {return []}
    const handle = typeof hold.handle === 'string' && hold.handle ? hold.handle : memberId
    const displayName = typeof hold.display_name === 'string' ? hold.display_name.trim() : ''

    return [{ memberId, handle, label: displayName || handle }]
  })
}

export function parseHostedRoom(raw: unknown, identity?: HostedRoomIdentity): HostedRoomSummary {
  const value = hostedRecord(raw)

  const room: HostedRoomSummary = {
    room_id: text(value.room_id),
    name: text(value.name),
    authority_gateway_id: text(value.authority_gateway_id),
    authority_epoch: integer(value.authority_epoch, 1),
    latest_seq: integer(value.latest_seq),
    members: []
  }

  if (identity && (identity.roomId !== room.room_id || identity.authorityGatewayId !== room.authority_gateway_id)) {
    throw new Error('Hosted room authority changed. Reopen it from its owning gateway; no local fallback is allowed.')
  }

  if (!Array.isArray(value.members)) {throw new Error('Invalid hosted room members')}
  room.members = value.members.map(rawMember => {
    const member = hostedRecord(rawMember)

    return {
      member_id: text(member.member_id), profile: text(member.profile), handle: text(member.handle),
      display_name: typeof member.display_name === 'string' ? member.display_name : undefined
    }
  })

  return room
}

/** Validate the entire page before committing it. Control events consume sequence
 * numbers too. A server's latest_seq is a high-water mark, NEVER a replay cursor. */
export function applyHostedPage(previous: HostedReplay, raw: unknown, room: HostedRoomSummary): HostedReplay & { hasMore: boolean } {
  const page = hostedRecord(raw)
  const authority = hostedRecord(page.authority)

  if (authority.gateway_id !== room.authority_gateway_id || authority.epoch !== room.authority_epoch) {
    throw new Error('Hosted room replay authority changed')
  }

  if (!Array.isArray(page.events)) {throw new Error('Invalid hosted room log')}
  let cursor = previous.cursor
  const ids = new Set(previous.events.map(event => event.event_id))

  const events = page.events.map(rawEvent => {
    const event = hostedRecord(rawEvent)
    const seq = integer(event.seq, 1)

    if (seq !== cursor + 1 || event.room_id !== room.room_id) {throw new Error('Hosted room log has a sequence gap or wrong room')}
    const id = text(event.event_id)

    if (ids.has(id)) {throw new Error('Hosted room log has a duplicate event')}
    ids.add(id)
    const actor = hostedRecord(event.actor)
    const payload = hostedRecord(event.payload)
    const kind = text(event.kind)

    if (kind === 'message.user' || kind === 'message.member') {
      text(payload.text)
      text(payload.thread_id)

      if (kind === 'message.member') {text(payload.member_id)}
    }

    if (typeof event.created_at !== 'number' || !Number.isFinite(event.created_at)) {throw new Error('Invalid hosted room timestamp')}
    const epoch = event.authority_epoch === null ? null : integer(event.authority_epoch, 1)

    if (epoch !== null && epoch > room.authority_epoch) {throw new Error('Hosted room event authority is ahead of state')}
    cursor = seq

    return {
      room_id: room.room_id, seq, event_id: id, kind,
      actor: { kind: text(actor.kind), id: text(actor.id), profile: typeof actor.profile === 'string' ? actor.profile : undefined },
      payload, created_at: event.created_at, authority_epoch: epoch
    }
  })

  const latest = integer(page.latest_seq)

  if (page.cursor !== cursor || latest < cursor || latest < room.latest_seq || page.has_more !== (cursor < latest) || (page.has_more && !events.length)) {
    throw new Error('Hosted room log returned an inconsistent cursor')
  }

  return { cursor, events: events.length ? [...previous.events, ...events] : previous.events, hasMore: page.has_more === true }
}

export function hostedMembers(room: HostedRoomSummary): GroupMember[] {
  // Member ids label the transcript; they are not Desktop profile/session routes.
  return room.members.map(member => ({ name: member.member_id, handle: member.handle, display_name: member.display_name || member.profile }))
}

export function hostedTranscript(replay: HostedReplay): GroupChat {
  const log: GroupMessage[] = replay.events.flatMap(event => {
    if (event.kind !== 'message.user' && event.kind !== 'message.member') {return []}

    return [{
      id: event.event_id, at: event.created_at * 1000,
      from: { kind: event.kind === 'message.user' ? 'user' as const : 'member' as const, name: event.kind === 'message.user' ? event.actor.id : String(event.payload.member_id) },
      text: String(event.payload.text), thread: String(event.payload.thread_id)
    }]
  })

  return { log, watermarks: {} }
}
