import { cleanup, render } from '@testing-library/react'
import type { ComponentProps } from 'react'
import { afterEach, expect, it, vi } from 'vitest'

vi.mock('@hermes/plugin-sdk', async () => {
  const { pluginSdkMock, createGroupGateway } = await import('./group-test-utils')

  return { ...await pluginSdkMock(createGroupGateway().host), Button: (p: ComponentProps<'button'>) => <button {...p} /> }
})

import { CanonicalGroupCycle } from './canonical-group-activity'
import { CANONICAL_GROUP_LOCALES } from './canonical-group-locales'
import { memberOutcomes, memberStatus, roomCycle, type RoomLogEvent } from './group-room-status'

afterEach(cleanup)

const members = [{ member_id: 'a', profile: 'a', handle: 'ada' }, { member_id: 'c', profile: 'c', handle: 'cy' }]

it('names only the running members of the thread whose round the strip shows', () => {
  const events: RoomLogEvent[] = [
    { seq: 1, event_id: 'a1', kind: 'message.user', payload: { text: 'go', thread_id: 'A' } },
    { seq: 2, kind: 'turn.settled', payload: { member_id: 'a', discussion_event_id: 'a1', thread_id: 'A', round_index: 0 } },
    { seq: 3, event_id: 'b1', kind: 'message.user', payload: { text: 'go', thread_id: 'B' } }
  ]

  // Thread B (focused) has only queued work while thread A's member c runs round 2.
  const activity = [
    { member_id: 'a', status: 'queued', thread_id: 'B', round_index: 0 },
    { member_id: 'c', status: 'running', thread_id: 'A', round_index: 1, started_at: 1 }
  ]

  expect(strip(events, activity)).toBe('Round 2 of 3 · Working: cy')
  // Both threads run: the focused thread's round, and only its own member beside it.
  expect(strip(events, [{ ...activity[0], status: 'running', started_at: 1 }, activity[1]])).toBe('Round 1 of 3 · Working: ada')
})

it.each([
  { pending_actions: [{ kind: 'retry' }] },
  { pending_actions: [{ kind: 'approval' }] },
  { pending_actions: [{ kind: 'discard' }] },
  { counts: { deferred: 1 } }
])('hides a settled footer until unresolved work is cleared: %j', driver => {
  const labels = CANONICAL_GROUP_LOCALES.en

  const events: RoomLogEvent[] = [
    { seq: 1, event_id: 'd', kind: 'message.user', payload: {} },
    { seq: 2, kind: 'turn.deferred', payload: { member_id: 'a', discussion_event_id: 'd', reason: 'member_unavailable' } },
    { seq: 3, kind: 'room.activity', payload: { discussion_event_id: 'd', status: 'settled' } }
  ]

  const props = { cycle: roomCycle(events, []), labels, members, statuses: new Map() }
  const view = render(<CanonicalGroupCycle {...props} driver={driver} />)
  expect(view.queryByRole('status')).toBeNull()
  view.rerender(<CanonicalGroupCycle {...props} driver={{ pending_actions: [], counts: { deferred: 0 } }} />)
  expect(view.getByRole('status').textContent).toBe(labels.cycleSettled)
  view.rerender(<CanonicalGroupCycle {...props} cycle={{ kind: 'working', round: 1, members: ['a'] }} driver={driver} />)
  expect(view.getByRole('status').textContent).toContain('Round 1')
})

function strip(events: RoomLogEvent[], activity: { member_id: string; status: string; thread_id: string; round_index: number }[]) {
  const outcomes = memberOutcomes(events)
  const statuses = new Map(members.map(m => [m.member_id, memberStatus(m.member_id, activity, outcomes)]))

  const view = render(<CanonicalGroupCycle cycle={roomCycle(events, activity, 3, 'B')} labels={{ ...CANONICAL_GROUP_LOCALES.en } as never}
    members={members} statuses={statuses} />)

  const text = view.getByRole('status').textContent
  view.unmount()

  return text
}
