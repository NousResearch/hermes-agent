import { expect, it } from 'vitest'

import {
  busyAt,
  type BusyMember,
  memberOutcomes,
  memberStatus,
  roomCycle,
  roomHandoffs,
  type RoomLogEvent,
  STALL_AFTER_S
} from './group-room-status'

const user = (seq: number, id: string): RoomLogEvent => ({
  seq,
  event_id: id,
  kind: 'message.user',
  payload: { text: 'go' }
})

const turn = (
  seq: number,
  kind: string,
  member: string,
  discussion: string,
  round: number,
  extra: Record<string, unknown> = {}
): RoomLogEvent => ({
  seq,
  kind,
  payload: { member_id: member, discussion_event_id: discussion, round_index: round, ...extra }
})

it('derives each member’s last outcome from the current discussion only', () => {
  const events = [
    user(1, 'd1'),
    turn(2, 'turn.failed', 'ada', 'd1', 0, { error: 'old' }),
    user(3, 'd2'),
    turn(4, 'turn.settled', 'ada', 'd2', 0, { passed: false }),
    turn(5, 'turn.settled', 'linus', 'd2', 0, { passed: true }),
    turn(6, 'turn.failed', 'grace', 'd2', 0, { error: 'HTTP 401: bad key\nstack…' }),
    turn(7, 'turn.failed', 'alan', 'd2', 0, { error: 'boom', reason_code: 'rate_limited' })
  ]

  const outcomes = memberOutcomes(events)
  expect(outcomes.get('ada')).toEqual({ kind: 'replied' })
  expect(outcomes.get('linus')).toEqual({ kind: 'passed' })
  expect(outcomes.get('grace')).toMatchObject({
    kind: 'failed',
    reason: 'HTTP 401: bad key',
    detail: 'HTTP 401: bad key\nstack…'
  })
  expect(outcomes.get('alan')?.reason).toBe('rate_limited')
  // A live task outranks a finished outcome.
  expect(memberStatus('ada', [{ member_id: 'ada', status: 'queued' }], outcomes).kind).toBe('queued')
})

it('reads a running member as stalled once its last sign of life is older than the threshold', () => {
  const now = 10_000
  const row = { member_id: 'ada', status: 'running', started_at: now - 600, tool: 'terminal' }

  const shown = (activity: Parameters<typeof memberStatus>[1]) =>
    busyAt(memberStatus('ada', activity, new Map()) as BusyMember, now)

  expect(shown([{ ...row, progress_at: now - 5 }])).toMatchObject({ kind: 'working', tool: 'terminal', elapsedS: 600 })
  expect(shown([{ ...row, progress_at: now - STALL_AFTER_S }]).kind).toBe('stalled')
  // Without progress frames (another process ran the turn) the start time is the sign of life.
  expect(shown([{ member_id: 'ada', status: 'running', started_at: now - 30 }]).kind).toBe('working')
  // The status itself is clock-free: the same rows give the same status whenever it is read.
  expect(memberStatus('ada', [row], new Map())).toEqual({
    kind: 'working',
    startedAt: now - 600,
    lastSignAt: now - 600,
    tool: 'terminal'
  })
})

it('tracks the round while members work and says how the discussion ended', () => {
  const events = [user(1, 'd'), turn(2, 'turn.settled', 'ada', 'd', 0), turn(3, 'turn.settled', 'linus', 'd', 0)]

  // Ada already answered round 0 and works again: round 2 of 3.
  expect(roomCycle(events, [{ member_id: 'ada', status: 'running' }], 3)).toEqual({
    kind: 'working',
    round: 2,
    maxRounds: 3,
    members: ['ada']
  })
  expect(roomCycle(events, [{ member_id: 'grace', status: 'running' }], null)).toEqual({
    kind: 'working',
    round: 1,
    maxRounds: undefined,
    members: ['grace']
  })

  const end = (payload: Record<string, unknown>) => [
    ...events,
    { seq: 9, kind: 'room.activity', payload: { discussion_event_id: 'd', ...payload } }
  ]
  expect(roomCycle(end({ status: 'settled' }), [], 3)).toEqual({ kind: 'settled' })
  expect(roomCycle(end({ status: 'bounded', reason_code: 'max_rounds' }), [], 3)).toEqual({ kind: 'bounded' })
  expect(roomCycle([...events, { seq: 9, kind: 'room.stop_requested', payload: {} }], [], 3)).toEqual({
    kind: 'stopRequested'
  })
  expect(roomCycle([], [], 3)).toBeNull()
})

it('shows a handoff only when the mentioned teammate then took a turn', () => {
  const members = [
    { member_id: 'ada', handle: 'ada' },
    { member_id: 'linus', handle: 'linus' },
    { member_id: 'grace', handle: 'grace' }
  ]

  const reply: RoomLogEvent = {
    seq: 2,
    kind: 'message.member',
    actor: { kind: 'member', id: 'ada' },
    payload: { text: 'Done; @linus please review. cc grace@example.com', discussion_event_id: 'd' }
  }

  expect(roomHandoffs([user(1, 'd'), reply, turn(3, 'turn.settled', 'linus', 'd', 1)], members)).toEqual([
    { afterSeq: 2, from: 'ada', to: 'linus' }
  ])
  expect(roomHandoffs([user(1, 'd'), reply], members)).toEqual([])
})
