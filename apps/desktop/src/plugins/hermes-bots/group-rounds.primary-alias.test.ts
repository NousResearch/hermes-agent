/**
 * Directional bot-to-bot handoff to the primary profile.
 *
 * The primary member's internal name is `default`; the user-facing alias is
 * `@hermes`. Live roster union rows and durable room descriptors often stamp
 * `handle: "default"` (same as the profile id). Using that handle as the
 * mention form shadows `botHandle()` and drops `@hermes`, so
 * code-farmer → @hermes never selects the primary bot while the reverse
 * direction still works (it matches `code-farmer` by name).
 */

import { beforeEach, describe, expect, it, vi } from 'vitest'

import { $groupChats } from './group-chat'
import { durableGroupChatMembers } from './group-membership'
import { parseGroupChatMentions, resolveGroupResponders, unaddressedGroupMentions } from './group-rounds'
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

const handleless: GroupMember[] = [
  { name: 'default', title: '' },
  { name: 'code-farmer', title: 'Code Farmer' }
]

const persisted: GroupMember[] = [
  { handle: 'default', name: 'default', title: '主要助理／任務協調者' },
  { handle: 'code-farmer', name: 'code-farmer', title: 'Code Farmer' }
]

const fromUser = (text: string): GroupMessage =>
  ({ at: 1, from: { kind: 'user', name: 'You' }, id: 'u1', text, thread: 't1' }) as GroupMessage

const fromMember = (name: string, text: string, id: string): GroupMessage =>
  ({ at: 2, from: { kind: 'member', name }, id, text, thread: 't1' }) as GroupMessage

beforeEach(() => {
  $groupChats.set({})
})

describe('primary @hermes alias in group mention parse', () => {
  it('resolves @hermes when the member has no precomputed handle', () => {
    const parsed = parseGroupChatMentions('@hermes Please reply with the letter B.', handleless)

    expect(parsed.mentioned.has('default')).toBe(true)
    expect(parsed.mentioned.size).toBe(1)
  })

  it('resolves @hermes when persist/union stamped handle: "default"', () => {
    const parsed = parseGroupChatMentions('@hermes Please reply with the letter B.', persisted)

    expect(parsed.mentioned.has('default')).toBe(true)
    expect(parsed.mentioned.size).toBe(1)
  })

  it('resolves @hermes after durableGroupChatMembers writes the room descriptor', () => {
    const durable = durableGroupChatMembers([
      { name: 'default', title: '主要助理／任務協調者' } as never,
      { name: 'code-farmer', title: 'Code Farmer' } as never
    ])

    expect(durable[0].handle).toBe('hermes')

    const parsed = parseGroupChatMentions('@hermes Please reply with the letter B.', durable)
    const keys = [...parsed.mentioned]

    expect(keys.some(key => String(key).includes('default'))).toBe(true)
  })

  it('still resolves a device-qualified handle and keeps @hermes as an alias', () => {
    const members: GroupMember[] = [
      { handle: 'default-vera', name: 'default' },
      { handle: 'code-farmer', name: 'code-farmer' }
    ]

    const byAlias = parseGroupChatMentions('@hermes take this', members)
    const byDevice = parseGroupChatMentions('@default-vera take this', members)

    expect(byAlias.mentioned.has('default')).toBe(true)
    expect(byDevice.mentioned.has('default')).toBe(true)
  })
})

describe('directional bot-to-bot continuation', () => {
  it('selects the primary bot when code-farmer hands off with @hermes', () => {
    const log = [
      fromUser('@code-farmer Please reply with one line only: mention Hermes and ask Hermes to reply with the letter B.'),
      fromMember('code-farmer', '@hermes Please reply with the letter B.', 'm1')
    ]
    const responders = resolveGroupResponders(log, persisted).map(member => member.name)

    $groupChats.set({ g: { log, members: persisted, roomId: 'room-1' } as never })

    expect(responders).toContain('default')
    expect(unaddressedGroupMentions('g', persisted, 't1')).toContain('default')
  })

  it('still selects code-farmer when the primary bot hands off the other way', () => {
    const log = [
      fromUser(
        '@hermes Please reply with one line only: mention Code Farmer and ask Code Farmer to reply with the letter D.'
      ),
      fromMember('default', '@code-farmer Please reply with the letter D.', 'm2')
    ]
    const responders = resolveGroupResponders(log, persisted).map(member => member.name)

    $groupChats.set({ g: { log, members: persisted, roomId: 'room-1' } as never })

    expect(responders).toContain('code-farmer')
    expect(unaddressedGroupMentions('g', persisted, 't1')).toContain('code-farmer')
  })
})
