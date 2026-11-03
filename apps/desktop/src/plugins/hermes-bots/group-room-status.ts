/**
 * What a gateway-owned room is doing, derived from the durable room log plus the live
 * `groups.state.driver_status` rows. Pure: the header chips, the typing rows, the cycle
 * footer and the Bots-list row all read these, so they can never disagree.
 */

/** One live task row from `driver_status.member_activity`; `tool`/`progress_at` only come
 *  from the process running the turn, so both are optional. */
export interface RoomMemberActivity {
  member_id: string
  status: string
  started_at?: number
  tool?: string
  progress_at?: number
}

export interface RoomLogEvent {
  seq: number
  event_id?: string
  kind: string
  payload: Record<string, unknown>
  actor?: { kind?: string; id?: string; display_name?: string }
  created_at?: number
}

export type MemberOutcomeKind = 'cancelled' | 'failed' | 'passed' | 'replied' | 'unavailable'

export interface MemberOutcome {
  kind: MemberOutcomeKind
  /** Short reason for a failure (reason_code, else the error's first line). */
  reason?: string
  /** Full error text, for a tooltip. */
  detail?: string
}

export type MemberStatus =
  | { kind: 'idle' }
  | { kind: 'queued' }
  | { kind: 'stalled' | 'working'; elapsedS?: number; quietS?: number; tool?: string }
  | ({ kind: MemberOutcomeKind } & Omit<MemberOutcome, 'kind'>)

/** A running member with no frame for this long reads as stalled. */
export const STALL_AFTER_S = 120

const TERMINAL_KINDS = new Set(['turn.settled', 'turn.failed', 'turn.cancelled', 'member.unavailable'])
const str = (value: unknown) => (typeof value === 'string' ? value : '')
const firstLine = (text: string) => text.split('\n').map(line => line.trim()).find(Boolean) ?? ''

/** The discussion the room is on now: the newest user message (its event id keys every
 *  turn it started). */
export function currentDiscussionId(events: readonly RoomLogEvent[]): string | undefined {
  for (let i = events.length - 1; i >= 0; i--) {
    const event = events[i]

    if (event.kind === 'message.user' || event.kind === 'message') {
      return event.event_id
    }
  }

  return undefined
}

const inDiscussion = (event: RoomLogEvent, discussion: string | undefined) =>
  !!discussion && str(event.payload.discussion_event_id) === discussion

const OUTCOME_OF: Record<string, (event: RoomLogEvent) => MemberOutcome> = {
  'turn.settled': event => ({ kind: event.payload.passed === true ? 'passed' : 'replied' }),
  'turn.failed': event => {
    const error = str(event.payload.error)

    return { kind: 'failed', reason: str(event.payload.reason_code) || firstLine(error) || undefined, detail: error || undefined }
  },
  'turn.cancelled': () => ({ kind: 'cancelled' }),
  'member.unavailable': event => ({ kind: 'unavailable', detail: str(event.payload.reason) || str(event.payload.error) || undefined })
}

/** Each member's last turn outcome in the current discussion (later turns win). */
export function memberOutcomes(events: readonly RoomLogEvent[]): Map<string, MemberOutcome> {
  const discussion = currentDiscussionId(events)
  const outcomes = new Map<string, MemberOutcome>()

  for (const event of events) {
    const member = str(event.payload.member_id)

    if (member && TERMINAL_KINDS.has(event.kind) && inDiscussion(event, discussion)) {
      outcomes.set(member, OUTCOME_OF[event.kind](event))
    }
  }

  return outcomes
}

/** One status per member: a live task (running outranks queued) beats the last outcome. */
export function memberStatus(
  memberId: string,
  activity: readonly RoomMemberActivity[] | undefined,
  outcomes: Map<string, MemberOutcome>,
  nowS: number
): MemberStatus {
  const rows = (activity ?? []).filter(row => row.member_id === memberId)
  const running = rows.find(row => row.status !== 'queued')

  if (running) {
    const elapsedS = typeof running.started_at === 'number' ? Math.max(0, nowS - running.started_at) : undefined
    const lastSign = running.progress_at ?? running.started_at
    const quietS = typeof lastSign === 'number' ? Math.max(0, nowS - lastSign) : undefined
    const kind = quietS !== undefined && quietS >= STALL_AFTER_S ? 'stalled' : 'working'

    return { kind, elapsedS, quietS, tool: running.tool || undefined }
  }

  if (rows.length) {
    return { kind: 'queued' }
  }

  const outcome = outcomes.get(memberId)

  return outcome ? { ...outcome } : { kind: 'idle' }
}

export const isBusyStatus = (status: MemberStatus) => status.kind === 'working' || status.kind === 'stalled'

/** `45s`, `1m20s`, `1h05m`. */
export function formatElapsed(seconds: number): string {
  const s = Math.floor(seconds)

  if (s < 60) {
    return `${s}s`
  }

  if (s < 3600) {
    return `${Math.floor(s / 60)}m${String(s % 60).padStart(2, '0')}s`
  }

  return `${Math.floor(s / 3600)}h${String(Math.floor((s % 3600) / 60)).padStart(2, '0')}m`
}

export type RoomCycle =
  | { kind: 'working'; round: number; maxRounds?: number }
  | { kind: 'bounded' | 'messageCap' | 'settled' | 'stopped' }

const ENDING: Record<string, (event: RoomLogEvent) => RoomCycle | null> = {
  'room.activity': event =>
    event.payload.status === 'settled'
      ? { kind: 'settled' }
      : event.payload.status === 'bounded'
        ? { kind: event.payload.reason_code === 'max_messages' ? 'messageCap' : 'bounded' }
        : null,
  'room.stop_requested': () => ({ kind: 'stopped' })
}

/**
 * Where the current discussion is: `Round N` (1-based) while members work, else how it ended.
 * The round is the newest `round_index` a turn of this discussion settled in, moved on by one
 * when a member already done in that round is working again.
 */
export function roomCycle(
  events: readonly RoomLogEvent[],
  activity: readonly RoomMemberActivity[] | undefined,
  maxRounds?: number | null
): RoomCycle | null {
  const discussion = currentDiscussionId(events)

  if (!discussion) {
    return null
  }

  const live = (activity ?? []).filter(row => row.status !== 'queued').map(row => row.member_id)
  const userAt = events.findIndex(event => event.event_id === discussion)
  let round = -1
  const doneInRound = new Map<number, Set<string>>()
  let ended: RoomCycle | null = null

  for (const event of events.slice(userAt + 1)) {
    const ending = ENDING[event.kind]?.(event)

    if (ending && (event.kind === 'room.stop_requested' || inDiscussion(event, discussion))) {
      ended = ending
    }

    if (TERMINAL_KINDS.has(event.kind) && inDiscussion(event, discussion)) {
      const index = Number(event.payload.round_index)

      if (Number.isInteger(index) && index >= 0) {
        round = Math.max(round, index)
        doneInRound.set(index, (doneInRound.get(index) ?? new Set()).add(str(event.payload.member_id)))
      }
    }
  }

  if (live.length || (activity ?? []).length) {
    const current = round < 0 ? 0 : live.some(id => doneInRound.get(round)?.has(id)) ? round + 1 : round
    const max = typeof maxRounds === 'number' && maxRounds > 0 ? maxRounds : undefined

    return { kind: 'working', round: current + 1, maxRounds: max }
  }

  return ended
}

export interface RoomHandoff {
  /** Insert after the event with this seq. */
  afterSeq: number
  from: string
  to: string
}

const mentions = (text: string, handle: string) =>
  !!handle && new RegExp(`(^|[^\\w@])@${handle.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}(?![\\w-])`, 'i').test(text)

/**
 * Visible handoffs: a member's reply that @mentions a teammate who then took a turn in the
 * same discussion. Display only, read from the log; routing is the gateway's business.
 */
export function roomHandoffs(
  events: readonly RoomLogEvent[],
  members: readonly { member_id: string; handle: string }[]
): RoomHandoff[] {
  const handoffs: RoomHandoff[] = []

  events.forEach((event, index) => {
    const from = event.kind === 'message.member' && event.actor?.kind === 'member' ? str(event.actor.id) : ''
    const text = str(event.payload.text) || str(event.payload.content)
    const discussion = str(event.payload.discussion_event_id)

    if (!from || !text) {
      return
    }

    for (const member of members) {
      if (member.member_id === from || !mentions(text, member.handle)) {
        continue
      }

      const tookTurn = events.slice(index + 1).some(later =>
        str(later.payload.member_id) === member.member_id && TERMINAL_KINDS.has(later.kind) &&
        (!discussion || str(later.payload.discussion_event_id) === discussion))

      if (tookTurn) {
        handoffs.push({ afterSeq: event.seq, from, to: member.member_id })
      }
    }
  })

  return handoffs
}
