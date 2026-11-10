/**
 * #94478 — the unresolved-handoff detector.
 *
 * A bot's @mention of a teammate inside its own reply IS visible to the next
 * round's responder selection, but the round loop exits before that round
 * runs: `spokeThisRound === 0` fires whenever the round's responders had no new
 * delta to read (everyone already spoke; the only new entry is the citing
 * reply), and a `GROUP_CHAT_MAX_*` cap can land in the same gap. The room then
 * records "settled" while a called bot never answered.
 *
 * `unaddressedGroupMentions` is what the quiet-round exit consults before
 * settling, so the continuation round it drives is only as correct as this
 * detector. Two subtleties it has to get right:
 *
 *  - **Log INDEX, not entry id, is the ordering.** Entry ids are UUIDs
 *    (`groupChatEntryId`), so "answered after the citing entry" can only mean
 *    positional order. Sorting by id marked answered handoffs as pending and
 *    re-drove them — a loop.
 *  - **A self-citation is not a handoff**, and neither is a user entry: a user
 *    send re-drives everyone anyway.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest'

import { $groupChats } from './group-chat'
import { unaddressedGroupMentions } from './group-rounds'
import type { GroupMember, GroupMessage } from './types'

vi.mock('@hermes/plugin-sdk', async () => {
  const { atom } = await import('nanostores')

  return {
    atom,
    host: {
      request: vi.fn(),
      state: { connectionId: { get: () => 'local' }, profile: { get: () => 'default' } }
    },
    queryClient: { invalidateQueries: vi.fn() },
    useQuery: vi.fn(),
    useValue: vi.fn()
  }
})

vi.mock('./shared', () => ({ getPluginCtx: () => null, ID: 'hermes-bots' }))

const members = [{ name: 'alpha' }, { name: 'beta' }, { name: 'gamma' }] as GroupMember[]

const from = (name: string, id: string, text: string): GroupMessage =>
  ({ at: 0, from: { kind: 'member', name }, id, text, thread: 't1' }) as GroupMessage

const fromUser = (id: string, text: string): GroupMessage =>
  ({ at: 0, from: { kind: 'user', name: 'You' }, id, text, thread: 't1' }) as GroupMessage

function room(log: GroupMessage[]) {
  $groupChats.set({ g: { log, members, roomId: 'room-1' } as never })
}

beforeEach(() => {
  $groupChats.set({})
})

describe('unaddressedGroupMentions', () => {
  it('flags a cited member with no later post', () => {
    room([from('alpha', 'zzzz-0002', 'hello'), from('beta', 'aaaa-0003', 'ping @gamma — take this')])

    expect(unaddressedGroupMentions('g', members, 't1')).toEqual(['gamma'])
  })

  it('counts a reply that follows in LOG order even when its id sorts lower', () => {
    room([from('beta', 'aaaa-0003', 'ping @gamma — take this'), from('gamma', '9999-0004', 'on it')])

    expect(unaddressedGroupMentions('g', members, 't1')).toEqual([])
  })

  it('re-flags a handoff cited again after the answer', () => {
    room([from('beta', 'a', 'ping @gamma'), from('gamma', 'b', 'on it'), from('beta', 'c', 'one more thing @gamma')])

    expect(unaddressedGroupMentions('g', members, 't1')).toEqual(['gamma'])
  })

  it('never treats a self-citation or a user entry as a handoff', () => {
    room([
      from('alpha', 'bbbb-0001', 'I will do @alpha things myself'),
      fromUser('cccc-0002', '@beta @gamma look here')
    ])

    expect(unaddressedGroupMentions('g', members, 't1')).toEqual([])
  })

  it('ignores citations in another thread', () => {
    const other = { ...from('beta', 'x', 'ping @gamma'), thread: 't2' } as GroupMessage

    room([other, from('alpha', 'y', 'unrelated')])

    expect(unaddressedGroupMentions('g', members, 't1')).toEqual([])
  })

  it('reports nothing for an unknown room', () => {
    expect(unaddressedGroupMentions('missing', members, 't1')).toEqual([])
  })
})
