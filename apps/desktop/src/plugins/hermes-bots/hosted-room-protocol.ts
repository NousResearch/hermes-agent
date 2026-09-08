import type { Attachment, AttachmentKind, GroupChat, GroupMember, GroupMessage } from './types'

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
  /** Whether this gateway both mints attachment ids and serves the byte RPCs. */
  attachments: boolean
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
    persistentHolds: features.includes('persistent_member_holds'),
    // Both halves are required: an older gateway may mint ids without serving the byte RPCs,
    // and a composer that offers files it cannot stage is worse than one that says it cannot.
    attachments: features.includes('attachment_ids') &&
      ['groups.attachment.put', 'groups.attachment.read'].every(m => methods.includes(m))
  }
}

const ATTACHMENT_ID = /^att_[0-9a-f]{32}$/
const ATTACHMENT_KINDS = new Set<AttachmentKind>(['file', 'image', 'pdf'])

/** The canonical attachment manifest one room message carries, as metadata only: the bytes
 * are fetched later through `groups.attachment.read`, so `data` stays empty here.
 *
 * Every committed field is validated, never coerced — `text` rejects a missing or empty
 * name/mime and `integer` rejects a missing, boolean, string or fractional size — because this
 * metadata is exactly what a later fetch is verified against. */
export function hostedAttachments(payload: Record<string, unknown>): Attachment[] {
  const raw = payload.attachments

  if (raw === undefined) {return []}

  if (!Array.isArray(raw)) {throw new Error('Invalid hosted room attachment manifest')}

  return raw.map(entry => {
    const value = hostedRecord(entry)
    const attachmentId = text(value.attachment_id)
    const kind = text(value.kind) as AttachmentKind

    if (!ATTACHMENT_ID.test(attachmentId) || !ATTACHMENT_KINDS.has(kind)) {
      throw new Error('Invalid hosted room attachment manifest')
    }

    return { attachmentId, data: '', kind, mime: text(value.mime), name: text(value.name), size: integer(value.size) }
  })
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

/** One action the owning gateway reports as waiting for the user. */
export interface HostedPendingAction {
  kind: 'approval' | 'retry'
  taskId: string
  memberId?: string
  executionGeneration?: number
  requestId?: string
  choices?: Array<'deny' | 'once'>
  command?: string
  reason?: string
}

/** Pending actions as the gateway reports them. A malformed or unknown row is dropped rather
 * than rendered: a control must carry the exact task, generation and approval it acts on, and
 * the backend's own bookkeeping (session ids) is never surfaced here. */
export function hostedPendingActions(driverStatus: unknown): HostedPendingAction[] {
  const status = driverStatus && typeof driverStatus === 'object' ? driverStatus as Record<string, unknown> : {}

  if (!Array.isArray(status.pending_actions)) {return []}

  return status.pending_actions.flatMap<HostedPendingAction>(raw => {
    if (!raw || typeof raw !== 'object') {return []}
    const row = raw as Record<string, unknown>
    const taskId = typeof row.task_id === 'string' ? row.task_id : ''

    if (!taskId) {return []}

    if (row.kind === 'retry') {return [{ kind: 'retry' as const, taskId }]}

    if (row.kind !== 'approval') {return []}
    const memberId = typeof row.member_id === 'string' ? row.member_id : ''
    const requestId = typeof row.request_id === 'string' ? row.request_id : ''
    const generation = row.execution_generation
    const approval = row.approval && typeof row.approval === 'object' ? row.approval as Record<string, unknown> : {}
    const command = typeof approval.command === 'string' ? approval.command : ''
    const reason = typeof approval.reason === 'string' ? approval.reason : ''

    const choices = (Array.isArray(approval.choices) ? approval.choices : [])
      .filter((choice): choice is 'deny' | 'once' => choice === 'once' || choice === 'deny')
      .filter(choice => choice === 'deny' || Boolean(command.trim()))

    // Generations start at 1; a zero or negative one cannot name a live attempt to approve.
    if (!memberId || !requestId || typeof generation !== 'number' || !Number.isSafeInteger(generation) ||
        generation < 1 || !choices.length) {
      return []
    }

    return [{ kind: 'approval', taskId, memberId, requestId, executionGeneration: generation, choices, command, reason }]
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
      text(payload.thread_id)
      const attachments = hostedAttachments(payload)

      if (kind === 'message.member') {
        text(payload.text)
        text(payload.member_id)
      } else if (typeof payload.text !== 'string' || (!payload.text && !attachments.length)) {
        // An attachment-only user message is valid and carries the empty string, exactly as
        // the gateway's own payload validation accepts it.
        throw new Error('Invalid hosted room message payload')
      }
    }

    // Validate imported records at admission too, before they can poison the rendered cache.
    if (kind === 'room.created') {legacyHistoryTranscript({ kind, payload, event_id: id })}

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

/** Imported records are a read-only archive, not native turns or planner input.
 * Keep source order and isolate thread/entry identities across independent histories. */
function legacyHistoryTranscript(event: Pick<HostedEvent, 'kind' | 'payload' | 'event_id'>): GroupMessage[] {
  if (event.kind !== 'room.created' || !('legacy_history' in event.payload)) {return []}
  const history = hostedRecord(event.payload.legacy_history)

  if (history.version !== 1 || !Array.isArray(history.sources)) {throw new Error('Invalid legacy history')}
  const surfaces = new Set<string>()
  const result: GroupMessage[] = []

  for (const rawSource of history.sources) {
    const source = hostedRecord(rawSource)
    const surface = text(source.surface)

    if (!['desktop', 'telegram'].includes(surface) || surfaces.has(surface)) {throw new Error('Invalid legacy history source')}
    surfaces.add(surface)
    text(source.room_name)
    const sessions = hostedRecord(source.sessions)

    if (!Array.isArray(source.records)) {throw new Error('Invalid legacy history records')}
    const ids = new Set<string>()

    for (const rawRecord of source.records) {
      const record = hostedRecord(rawRecord)
      const author = hostedRecord(record.from)
      const id = text(record.id)
      const name = text(author.name)

      if (ids.has(id) || !['user', 'member'].includes(String(author.kind)) || typeof record.text !== 'string' ||
          typeof record.at !== 'number' || !Number.isFinite(record.at) || ('images' in record && (!Array.isArray(record.images) || record.images.length))) {
        throw new Error('Invalid legacy history record')
      }

      ids.add(id)

      if (author.kind === 'member') {
        const session = hostedRecord(sessions[name])
        text(session.id)
        text(session.title)
      }

      const provenance = surface === 'desktop' ? 'Desktop' : 'Telegram'

      result.push({
        id: JSON.stringify([event.event_id, surface, id]), at: record.at,
        from: { kind: author.kind as 'user' | 'member', name, source: `Imported ${provenance}` },
        text: `[Imported ${provenance} history, not a native turn]\n\n${record.text}`,
        thread: JSON.stringify(['legacy-history', surface, record.thread ?? 'legacy'])
      })
    }
  }

  if (surfaces.size !== 2) {throw new Error('Both legacy history sources are required')}

  return result
}

export function hostedTranscript(replay: HostedReplay): GroupChat {
  const log: GroupMessage[] = replay.events.flatMap(event => {
    if (event.kind === 'room.created') {return legacyHistoryTranscript(event)}

    if (event.kind !== 'message.user' && event.kind !== 'message.member') {return []}

    const images = hostedAttachments(event.payload)

    return [{
      id: event.event_id, at: event.created_at * 1000,
      from: { kind: event.kind === 'message.user' ? 'user' as const : 'member' as const, name: event.kind === 'message.user' ? event.actor.id : String(event.payload.member_id) },
      text: String(event.payload.text), thread: String(event.payload.thread_id),
      ...(images.length ? { images } : {})
    }]
  })

  return { log, watermarks: {} }
}
