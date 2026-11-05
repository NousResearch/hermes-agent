import { expect, it } from 'vitest'

import { focusThread, memberOutcomes, memberStatus, roomCycle, type RoomLogEvent } from './group-room-status'

const user = (seq: number, id: string, thread: string): RoomLogEvent => ({
  seq,
  event_id: id,
  kind: 'message.user',
  payload: { text: 'go', thread_id: thread }
})

const turn = (
  seq: number,
  kind: string,
  member: string,
  discussion: string,
  thread: string,
  round: number,
  extra: Record<string, unknown> = {}
): RoomLogEvent => ({
  seq,
  kind,
  payload: { member_id: member, discussion_event_id: discussion, thread_id: thread, round_index: round, ...extra }
})

// Thread A: a replied, b failed, c still works in round 2; then the user posts in thread B.
const events = [
  user(1, 'a1', 'A'),
  turn(2, 'turn.settled', 'a', 'a1', 'A', 0),
  turn(3, 'turn.failed', 'b', 'a1', 'A', 0, { error: 'upstream 502', reason_code: 'provider_error' }),
  turn(4, 'turn.settled', 'c', 'a1', 'A', 0),
  user(5, 'b1', 'B')
]

const activity = [{ member_id: 'c', status: 'running', thread_id: 'A', round_index: 1 }]

it('keeps a member’s outcome from a thread that still works after the user posts in another thread', () => {
  const outcomes = memberOutcomes(events)

  expect(focusThread(events)).toBe('B')
  expect(memberStatus('b', activity, outcomes, 0)).toMatchObject({ kind: 'failed', reason: 'provider_error' })
  expect(memberStatus('a', activity, outcomes, 0).kind).toBe('replied')
  expect(memberStatus('c', activity, outcomes, 0).kind).toBe('working')
  // Once thread A is done, its failure still stands while thread B runs; a newer message in A supersedes it.
  expect(
    memberStatus('b', [{ member_id: 'a', status: 'running', thread_id: 'B' }], memberOutcomes(events), 0).kind
  ).toBe('failed')
  expect(memberOutcomes([...events, user(6, 'a2', 'A')]).get('b')).toBeUndefined()
})

it('takes the round from the live task’s own thread, never the newest thread’s', () => {
  expect(roomCycle(events, activity, 3, focusThread(events))).toEqual({
    kind: 'working',
    round: 2,
    maxRounds: 3,
    members: ['c']
  })
  // The thread being replied in wins when it has live work of its own.
  const both = [...activity, { member_id: 'a', status: 'running', thread_id: 'B', round_index: 0 }]
  expect(roomCycle(events, both, 3, 'B')).toEqual({ kind: 'working', round: 1, maxRounds: 3, members: ['a'] })
  // A focused thread with only queued work does not lend its round to another thread's running member.
  const queuedFocus = [...activity, { member_id: 'a', status: 'queued', thread_id: 'B', round_index: 0 }]
  expect(roomCycle(events, queuedFocus, 3, 'B')).toEqual({ kind: 'working', round: 2, maxRounds: 3, members: ['c'] })
})

it('reads a member the driver parked as unavailable, and never a room-level stall', () => {
  const parked = [
    user(1, 'd', 'T'),
    turn(2, 'turn.deferred', 'scout', 'd', 'T', 0, { execution_generation: 1, reason: 'member_unavailable' }),
    { seq: 3, kind: 'member.unavailable', payload: { error: 'prepare failed', reason_code: 'prepare_failed' } }
  ]

  const outcomes = memberOutcomes(parked)
  expect(memberStatus('scout', [], outcomes, 0).kind).toBe('unavailable')
  expect([...outcomes.keys()]).toEqual(['scout'])
})
