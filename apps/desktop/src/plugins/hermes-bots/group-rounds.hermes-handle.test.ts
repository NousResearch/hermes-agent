/**
 * A room holding the default bot and a real (pre-reservation) `hermes`
 * profile: the stored member descriptors must keep the roster's claim, or
 * every reader that re-derives handles from them (the turn prompt, mention
 * parsing, the next re-persist) tells both bots they are @hermes.
 */

import { beforeEach, expect, it, vi } from 'vitest'

import { botMentionTag, claimDefaultHandles } from './data'
import { $groupChats } from './group-chat'
import { durableGroupChatMembers, groupMemberKey } from './group-membership'
import { buildGroupChatTurnPrompt } from './group-round-prompt'
import { parseGroupChatMentions } from './group-rounds'
import type { RosterRow } from './types'

vi.mock('@hermes/plugin-sdk', async () => {
  const { atom } = await import('nanostores')

  return {
    atom,
    host: { request: vi.fn(), state: { connectionId: { get: () => 'local' }, profile: { get: () => 'default' } } },
    queryClient: { invalidateQueries: vi.fn() },
    useQuery: vi.fn(),
    useValue: vi.fn()
  }
})
vi.mock('./shared', () => ({ getPluginCtx: () => null, ID: 'hermes-bots' }))

beforeEach(() => $groupChats.set({}))

it('keeps @default and @hermes apart in a room whose live roster rows are gone', () => {
  const live = claimDefaultHandles<RosterRow>(
    ['default', 'hermes', 'researcher'].map(name => ({ connectionId: 'local', name }))
  )

  const stored = durableGroupChatMembers(live)
  // The next send re-persists from whatever is seated — the stored rows here.
  const repersisted = durableGroupChatMembers(stored as RosterRow[])

  for (const members of [stored, repersisted]) {
    expect(members.map(member => botMentionTag(member))).toEqual(['default', 'hermes', 'researcher'])
    expect([...parseGroupChatMentions('@hermes please review', members).mentioned]).toEqual([
      groupMemberKey(members[1])
    ])
    expect([...parseGroupChatMentions('@default please review', members).mentioned]).toEqual([
      groupMemberKey(members[0])
    ])

    const header = (viewer: number) =>
      buildGroupChatTurnPrompt({ groupName: 'g', members, viewer: members[viewer], deltaLines: [] }).split('\n')[0]

    expect(header(0)).toMatch(
      /You are @default, one participant in a group chat with @hermes \[on local\], @researcher/
    )
    expect(header(1)).toMatch(
      /You are @hermes, one participant in a group chat with @default \[on local\], @researcher/
    )
  }
})
