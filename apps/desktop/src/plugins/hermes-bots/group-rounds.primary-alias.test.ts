import { beforeEach, expect, it, vi } from 'vitest'

import { $groupChats } from './group-chat'
import { durableGroupChatMembers, groupMemberKey } from './group-membership'
import { parseGroupChatMentions, resolveGroupResponders, unaddressedGroupMentions } from './group-rounds'
import type { GroupMember, GroupMessage } from './types'

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

it('keeps primary handoffs callable in both directions across persisted descriptor shapes', () => {
  for (const handle of [undefined, 'default', 'hermes']) {
    const original: GroupMember[] = [{ name: 'default', handle }, { name: 'code-farmer' }]

    const durable = durableGroupChatMembers(original)
    expect(durable.map(member => member.handle)).toEqual(['hermes', 'code-farmer'])

    for (const members of [original, durable]) {
      for (const [sender, target, senderTag, targetTag] of [
        ['code-farmer', 'default', 'code-farmer', 'hermes'],
        ['default', 'code-farmer', 'hermes', 'code-farmer']
      ]) {
        const targetKey = groupMemberKey(members.find(member => member.name === target)!)

        const log: GroupMessage[] = [
          { at: 1, from: { kind: 'user', name: 'You' }, id: 'u', text: `@${senderTag} begin`, thread: 't' },
          { at: 2, from: { kind: 'member', name: sender }, id: 'm', text: `@${targetTag} continue`, thread: 't' }
        ]

        $groupChats.set({ g: { log, members, roomId: 'room', watermarks: {} } })
        expect([...parseGroupChatMentions(log[1].text, members).mentioned]).toEqual([targetKey])
        expect(resolveGroupResponders(log, members).map(member => member.name)).toContain(target)
        expect(unaddressedGroupMentions('g', members, 't')).toEqual([targetKey])
      }
    }
  }
})

it('keeps qualified remote defaults distinct from the primary alias regardless of roster order', () => {
  const local: GroupMember = { name: 'default', handle: 'hermes', connectionId: 'local', sourceScoped: true }

  const remotes: GroupMember[] = ['vera', 'spark'].map(device => ({
    name: 'default',
    handle: `default-${device}`,
    connectionId: device,
    remoteSource: true,
    sourceScoped: true
  }))

  for (const members of [[local, ...remotes], [...remotes].reverse().concat(local)]) {
    for (const descriptors of [members, durableGroupChatMembers(members)]) {
      for (const [tag, key] of [
        ['hermes', 'local::default'],
        ['default-vera', 'vera::default'],
        ['default-spark', 'spark::default']
      ]) {
        expect([...parseGroupChatMentions(`@${tag} continue`, descriptors).mentioned]).toEqual([key])
      }
    }
  }
})
