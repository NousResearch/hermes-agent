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
  /** The thread and 0-based round the task works in (newer gateways only). */
  thread_id?: string
  round_index?: number
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

/** A running member, in absolute times (epoch seconds) so it does not change as the clock ticks: the
 *  chip that shows it reads the clock (`busyAt`), the room's derivations never do. */
export interface BusyMember { kind: 'working'; startedAt?: number; lastSignAt?: number; tool?: string }

export type MemberStatus =
  | { kind: 'idle' }
  | { kind: 'queued' }
  | BusyMember
  | ({ kind: MemberOutcomeKind } & Omit<MemberOutcome, 'kind'>)

/** What a busy chip shows at `nowS`: working or stalled (no sign of life for `STALL_AFTER_S`), and times. */
export type BusyStatus = { kind: 'stalled' | 'working'; elapsedS?: number; quietS?: number; tool?: string }

/** A running member with no frame for this long reads as stalled. */
export const STALL_AFTER_S = 120

const TERMINAL_KINDS = new Set(['turn.settled', 'turn.failed', 'turn.cancelled', 'turn.deferred'])
const str = (value: unknown) => (typeof value === 'string' ? value : '')
const firstLine = (text: string) => text.split('\n').map(line => line.trim()).find(Boolean) ?? ''
const threadOf = (event: RoomLogEvent) => str(event.payload.thread_id)
const isUserMessage = (event: RoomLogEvent) => event.kind === 'message.user' || event.kind === 'message'

/** Each thread's current discussion: its newest user message (that event id keys every turn it started). */
export function threadDiscussions(events: readonly RoomLogEvent[]): Map<string, string> {
  const discussions = new Map<string, string>()

  for (const event of events) {
    if (isUserMessage(event) && event.event_id) {
      discussions.set(threadOf(event), event.event_id)
    }
  }

  return discussions
}

/** The thread the room is read in: the one being replied in, else the newest user message's. */
export function focusThread(events: readonly RoomLogEvent[], replyThread?: string): string | undefined {
  if (replyThread) {
    return replyThread
  }

  for (let i = events.length - 1; i >= 0; i--) {
    if (isUserMessage(events[i])) {
      return threadOf(events[i])
    }
  }

  return undefined
}

const OUTCOME_OF: Record<string, (event: RoomLogEvent) => MemberOutcome | null> = {
  'turn.settled': event => ({ kind: event.payload.passed === true ? 'passed' : 'replied' }),
  'turn.failed': event => {
    const error = str(event.payload.error)

    return { kind: 'failed', reason: str(event.payload.reason_code) || firstLine(error) || undefined, detail: error || undefined }
  },
  'turn.cancelled': () => ({ kind: 'cancelled' }),
  // The driver parks a member it cannot reach; the room-level `member.unavailable` names no member.
  'turn.deferred': event => (event.payload.reason === 'member_unavailable' ? { kind: 'unavailable' } : null)
}

/**
 * Each member's last turn outcome, read per thread: a turn counts while it belongs to its own thread's
 * current discussion (later turns win). A newer message in that thread supersedes it; a message in another
 * thread does not, so a failure in one thread stays visible while another thread runs.
 */
export function memberOutcomes(events: readonly RoomLogEvent[]): Map<string, MemberOutcome> {
  const discussions = threadDiscussions(events)
  const outcomes = new Map<string, MemberOutcome>()

  for (const event of events) {
    const member = str(event.payload.member_id)
    const thread = threadOf(event)

    if (!member || !TERMINAL_KINDS.has(event.kind) || str(event.payload.discussion_event_id) !== discussions.get(thread)) {
      continue
    }

    const outcome = OUTCOME_OF[event.kind](event)

    if (outcome) {
      outcomes.set(member, outcome)
    }
  }

  return outcomes
}

/** One status per member: a live task (running outranks queued) beats the last outcome. Clock-free. */
export function memberStatus(
  memberId: string,
  activity: readonly RoomMemberActivity[] | undefined,
  outcomes: Map<string, MemberOutcome>
): MemberStatus {
  const rows = (activity ?? []).filter(row => row.member_id === memberId)
  const running = rows.find(row => row.status !== 'queued')

  if (running) {
    return { kind: 'working', startedAt: running.started_at, lastSignAt: running.progress_at ?? running.started_at,
      tool: running.tool || undefined }
  }

  if (rows.length) {
    return { kind: 'queued' }
  }

  const outcome = outcomes.get(memberId)

  return outcome ? { ...outcome } : { kind: 'idle' }
}

export const isBusyStatus = (status: MemberStatus): status is BusyMember => status.kind === 'working'

/** A busy member as its chip shows it at `nowS`. */
export function busyAt(status: BusyMember, nowS: number): BusyStatus {
  const elapsedS = typeof status.startedAt === 'number' ? Math.max(0, nowS - status.startedAt) : undefined
  const quietS = typeof status.lastSignAt === 'number' ? Math.max(0, nowS - status.lastSignAt) : undefined

  return { kind: quietS !== undefined && quietS >= STALL_AFTER_S ? 'stalled' : 'working', elapsedS, quietS, tool: status.tool }
}

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
  | { kind: 'working'; round: number; maxRounds?: number; /** The live members of the thread `round` is from. */ members: string[] }
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

const ended = (events: readonly RoomLogEvent[], discussion: string | undefined): RoomCycle | null => {
  const userAt = events.findIndex(event => !!discussion && event.event_id === discussion)
  let result: RoomCycle | null = null

  if (userAt < 0) {
    return null
  }

  for (const event of events.slice(userAt + 1)) {
    const ending = ENDING[event.kind]?.(event)

    if (ending && (event.kind === 'room.stop_requested' || str(event.payload.discussion_event_id) === discussion)) {
      result = ending
    }
  }

  return result
}

/** Older gateways send no `round_index`: the newest round a turn of this discussion finished in, moved on
 *  by one when a member already done in that round is working again. */
function guessedRound(events: readonly RoomLogEvent[], discussion: string | undefined, live: readonly string[]): number {
  let round = -1
  const doneInRound = new Map<number, Set<string>>()

  for (const event of events) {
    const index = Number(event.payload.round_index)

    if (TERMINAL_KINDS.has(event.kind) && !!discussion && str(event.payload.discussion_event_id) === discussion &&
        Number.isInteger(index) && index >= 0) {
      round = Math.max(round, index)
      doneInRound.set(index, (doneInRound.get(index) ?? new Set()).add(str(event.payload.member_id)))
    }
  }

  return round < 0 ? 0 : live.some(id => doneInRound.get(round)?.has(id)) ? round + 1 : round
}

/**
 * Where the room is: `Round N` (1-based) while members work, else how the focused thread's discussion
 * ended. The round is the live tasks' own (the focused thread's when it has any, else whichever thread
 * works), never another thread's.
 */
export function roomCycle(
  events: readonly RoomLogEvent[],
  activity: readonly RoomMemberActivity[] | undefined,
  maxRounds?: number | null,
  focus: string | undefined = focusThread(events)
): RoomCycle | null {
  const discussions = threadDiscussions(events)

  if (!discussions.size) {
    return null
  }

  const rows = activity ?? []

  if (!rows.length) {
    return ended(events, focus === undefined ? undefined : discussions.get(focus))
  }

  // One thread owns the strip, so its round and the names beside it agree: the focused thread when it has
  // running work, else a thread that does, else the focused thread's queued work.
  const isLive = (row: RoomMemberActivity) => row.status !== 'queued'
  const inThread = (thread: string | undefined) => rows.filter(row => row.thread_id !== undefined && row.thread_id === thread)
  const inFocus = inThread(focus)
  const running = rows.find(row => isLive(row) && row.thread_id !== undefined)?.thread_id
  const owning = inFocus.some(isLive) ? inFocus : running !== undefined ? inThread(running) : inFocus.length ? inFocus : rows
  const rounds = owning.map(row => row.round_index).filter((index): index is number => typeof index === 'number' && Number.isInteger(index) && index >= 0)
  const thread = owning.find(row => row.thread_id !== undefined)?.thread_id ?? focus
  const live = owning.filter(isLive).map(row => row.member_id)
  const current = rounds.length ? Math.max(...rounds) : guessedRound(events, thread === undefined ? undefined : discussions.get(thread), live)
  const max = typeof maxRounds === 'number' && maxRounds > 0 ? maxRounds : undefined

  return { kind: 'working', round: current + 1, maxRounds: max, members: live }
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
 * One pass over the log: each (member, discussion) keeps the index of its last turn.
 */
export function roomHandoffs(
  events: readonly RoomLogEvent[],
  members: readonly { member_id: string; handle: string }[]
): RoomHandoff[] {
  const lastTurn = new Map<string, number>()

  events.forEach((event, index) => {
    const member = str(event.payload.member_id)

    if (member && TERMINAL_KINDS.has(event.kind)) {
      lastTurn.set(JSON.stringify([member, str(event.payload.discussion_event_id)]), index)
      lastTurn.set(JSON.stringify([member, null]), index)
    }
  })

  const handoffs: RoomHandoff[] = []

  events.forEach((event, index) => {
    const from = event.kind === 'message.member' && event.actor?.kind === 'member' ? str(event.actor.id) : ''
    const text = str(event.payload.text) || str(event.payload.content)
    const discussion = str(event.payload.discussion_event_id)

    if (!from || !text) {
      return
    }

    for (const member of members) {
      const turnAt = lastTurn.get(JSON.stringify([member.member_id, discussion || null]))

      if (member.member_id !== from && turnAt !== undefined && turnAt > index && mentions(text, member.handle)) {
        handoffs.push({ afterSeq: event.seq, from, to: member.member_id })
      }
    }
  })

  return handoffs
}
