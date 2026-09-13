import { beforeEach, expect, it, vi } from 'vitest'

const fixture = vi.hoisted(() => ({
  listeners: new Map<string, (event: unknown) => void>(),
  read: vi.fn(),
  release: vi.fn(),
  active: 'source-A'
}))

vi.mock('@hermes/plugin-sdk', async () => {
  const { pluginSdkMock, createGroupGateway } = await import('./group-test-utils')
  const base = await pluginSdkMock(createGroupGateway().host)

  return {
    ...base,
    host: {
      ...base.host,
      activeConnectionId: () => fixture.active,
      listReviewSummaries: fixture.read,
      retainProfile: vi.fn(async () => fixture.release),
      onEvent: (type: string, handler: (event: unknown) => void) => {
        fixture.listeners.set(type, handler)

        return () => fixture.listeners.delete(type)
      }
    }
  }
})
import { $groupChats, durableGroupChatRooms } from './group-chat'
import { groupMemberKey } from './group-membership'
import { mergeGroupReviewReceipts, readGroupReviewReceipts, watchGroupReviewReceipts } from './group-review-receipts'
import type { GroupMember } from './types'

const tick = async () => {
  for (let i = 0; i < 12; i++) {
    await Promise.resolve()
  }
}

const row = (id: string, text: string, timestamp = 10) => ({
  content: text,
  display_kind: 'review_summary',
  display_metadata: { review_id: id },
  timestamp
})

const member: GroupMember = { name: 'Builder' }
beforeEach(() => {
  fixture.read.mockReset().mockResolvedValue({ messages: [] })
  fixture.release.mockReset()
  fixture.listeners.clear()
  fixture.active = 'source-A'
  $groupChats.set({ Room: { roomId: 'room-1', log: [], watermarks: {}, sessions: { Builder: 'hidden-group' } } })
})

it('backfills hidden-session receipts, deduplicates live/reopen, and leaves the model log untouched', async () => {
  fixture.read.mockResolvedValue({ messages: [row('r1', 'Saved group technique')] })
  let stop = watchGroupReviewReceipts('Room', [member])
  await tick()
  expect($groupChats.get().Room.reviewReceipts).toHaveLength(1)
  expect($groupChats.get().Room.reviewReceipts?.[0].at).toBe(10000)
  fixture.active = 'source-B'
  // An unrelated PRIVATE Bot Chat event is just an invalidation. Its text
  // must never be copied; only the exact captured hidden-session read counts.
  fixture.listeners.get('review.summary')?.({ connectionId: 'source-B', payload: { text: 'PRIVATE' } })
  await tick()
  expect(
    fixture.read.mock.calls.every(([route, sid]) => route.connectionId === 'source-A' && sid === 'hidden-group')
  ).toBe(true)
  expect(JSON.stringify($groupChats.get().Room.reviewReceipts)).not.toContain('PRIVATE')
  expect($groupChats.get().Room.log).toEqual([])
  expect($groupChats.get().Room.watermarks).toEqual({})
  expect(durableGroupChatRooms().Room.reviewReceipts).toHaveLength(1)
  stop()
  expect(fixture.release).toHaveBeenCalledTimes(1)
  fixture.active = 'source-A'
  stop = watchGroupReviewReceipts('Room', [member])
  await tick()
  expect($groupChats.get().Room.reviewReceipts).toHaveLength(1)
  stop()
})

it('qualifies same-named members by connection and keeps aliases on their backend target', async () => {
  const members: GroupMember[] = ['A', 'B'].map(connectionId => ({
    name: 'Builder',
    sourceScoped: true,
    connectionId,
    route: { connectionId, mode: 'remote', profile: 'Builder', targetProfile: 'actual-profile' }
  }))

  $groupChats.set({
    Room: {
      roomId: 'room-1',
      log: [],
      watermarks: {},
      sessions: Object.fromEntries(members.map(m => [groupMemberKey(m), `hidden-${m.connectionId}`]))
    }
  })
  fixture.read.mockImplementation(async (route, sid) => ({
    messages: [row('same-id', `${route.connectionId}:${sid}`)]
  }))
  const stop = watchGroupReviewReceipts('Room', members)
  await tick()
  const receipts = $groupChats.get().Room.reviewReceipts || []
  expect(receipts).toHaveLength(2)
  expect(new Set(receipts.map(r => r.id)).size).toBe(2)
  expect(fixture.read.mock.calls.every(([route]) => route.targetProfile === 'actual-profile')).toBe(true)
  stop()
})

it('follows a renamed room but discards reads after disband and same-name recreation', async () => {
  let resolve!: (result: { messages: unknown[] }) => void
  fixture.read.mockReturnValue(
    new Promise(r => {
      resolve = r
    })
  )
  let stop = watchGroupReviewReceipts('Room', [member])
  await tick()
  $groupChats.set({ Renamed: $groupChats.get().Room })
  resolve({ messages: [row('r1', 'rename receipt')] })
  await tick()
  expect($groupChats.get().Renamed.reviewReceipts).toHaveLength(1)
  stop()
  fixture.read.mockReturnValue(
    new Promise(r => {
      resolve = r
    })
  )
  stop = watchGroupReviewReceipts('Renamed', [member])
  await tick()
  $groupChats.set({})
  $groupChats.set({ Renamed: { roomId: 'different', log: [], watermarks: {}, sessions: { Builder: 'hidden-group' } } })
  resolve({ messages: [row('r2', 'must be discarded')] })
  await tick()
  expect($groupChats.get().Renamed.reviewReceipts).toBeUndefined()
  stop()
})

it('bounds and validates the display cache without classifying ordinary prose as reviews', () => {
  const rows = Array.from({ length: 80 }, (_, i) => row(`r${i}`, 'saved', i + 1))
  const receipts = readGroupReviewReceipts(
    [...rows, { content: 'Self-improvement review' }, row('bad', 'bad', NaN)],
    member,
    'owner'
  )
  const bounded = mergeGroupReviewReceipts([], receipts)
  expect(bounded).toHaveLength(50)
  expect(bounded[0].at).toBe(31000)
  expect(mergeGroupReviewReceipts(bounded, bounded)).toEqual(bounded)
})
