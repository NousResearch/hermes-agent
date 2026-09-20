import { beforeEach, describe, expect, it, vi } from 'vitest'

import type * as data from './data'
import type * as groupChat from './group-chat'
import type * as members from './group-chat-view-members'
import type * as groupMembership from './group-membership'
import { createGroupGateway, runTimersInline, scriptedStorage } from './group-test-utils'
import type { RosterRow } from './types'

// Both membership doors (room-side "Manage members", per-Bot "Manage groups")
// must leave the room reading the SAME roster from its two sources — local
// groups[] metadata and stored member descriptors — and a removed Bot must
// stay removed through the next gateway mirror poll.

const { host } = vi.hoisted(() => ({ host: {} as Record<string, unknown> }))

vi.mock('@hermes/plugin-sdk', async () => {
  const { pluginSdkMock } = await import('./group-test-utils')

  return pluginSdkMock(host)
})

interface Room {
  chat: typeof groupChat
  data: typeof data
  members: typeof members
  membership: typeof groupMembership
}

async function loadRoom(): Promise<Room> {
  vi.resetModules()
  const gateway = createGroupGateway()

  for (const key of Object.keys(host)) {
    delete host[key]
  }

  Object.assign(host, gateway.host)

  const [chat, d, m, membership, shared] = await Promise.all([
    import('./group-chat'),
    import('./data'),
    import('./group-chat-view-members'),
    import('./group-membership'),
    import('./shared')
  ])

  shared.setPluginCtx(scriptedStorage(gateway.storage))

  return { chat, data: d, members: m, membership }
}

const programmer: RosterRow = { name: 'programmer' }
const planner: RosterRow = { name: 'planner' }
const reviewer: RosterRow = { name: 'reviewer' }

const seatedNames = (room: Room) =>
  room.membership
    .groupChatMemberBots('Core', room.data.$lastRoster.get(), room.data.$botMeta.get())
    .map(bot => bot.name)
    .sort()

function seedCore(room: Room, extra: Partial<groupChat.GroupChatRoom> = {}) {
  room.data.$lastRoster.set([programmer, planner, reviewer])
  room.data.$botMeta.set({ programmer: { groups: ['Core'] }, reviewer: { groups: ['Core'] } })
  room.chat.updateGroupChat(
    'Core',
    r => ({
      ...r,
      members: room.membership.durableGroupChatMembers([programmer, reviewer]),
      syncRevision: 7,
      ...extra
    }),
    { sync: false }
  )
}

beforeEach(() => {
  runTimersInline()
})

describe('setGroupChatMembers', () => {
  it("outranks a tie-revision mirror poll and clears the removed member's room state", async () => {
    const room = await loadRoom()
    seedCore(room, {
      holds: { reviewer: { at: 1 } },
      stranded: { reviewer: { before: 0, thread: 't1' } },
      sessions: { 'thread:t1::reviewer': 'sid-rev', 'thread:t1::programmer': 'sid-prog' },
      watermarks: { 't1::reviewer': 1, 't1::programmer': 1 }
    })

    await room.members.setGroupChatMembers('Core', [programmer, planner])

    // The gateway projection still carries the pre-save roster at the SAME
    // revision the room had before Save (the poll that raced the publish).
    const mirror = {
      version: 3,
      rooms: {
        'name:Core': { name: 'Core', log: [], members: [{ name: 'programmer' }, { name: 'reviewer' }], revision: 7 }
      },
      deleted: {}
    }

    room.chat.$groupChats.set(
      room.chat.mergeRemoteGroupChatSnapshotIntoRooms(mirror as never, room.chat.$groupChats.get())
    )

    expect(seatedNames(room)).toEqual(['planner', 'programmer'])
    const saved = room.chat.$groupChats.get().Core as groupChat.GroupChatRoom
    expect(saved.syncRevision).toBe(8)
    expect(saved.holds).toEqual({})
    expect(saved.stranded).toEqual({})
    expect(saved.sessions).toEqual({ 'thread:t1::programmer': 'sid-prog' })
    expect(saved.watermarks).toEqual({ 't1::programmer': 1 })
  })

  it('keeps a same-named local Bot out when selecting only its Connection counterpart', async () => {
    const room = await loadRoom()
    const local: RosterRow = { name: 'planner', title: 'Local Planner' }

    const remote: RosterRow = {
      connectionId: 'remote-1',
      connectionKind: 'remote',
      name: 'planner',
      remoteSource: true,
      route: { connectionId: 'remote-1', mode: 'remote', profile: 'planner', targetProfile: 'planner' },
      sourceScoped: true,
      title: 'Remote Planner'
    }

    room.data.$lastRoster.set([local, remote, reviewer])
    room.data.$botMeta.set({ planner: { groups: ['Core'] } })

    await room.members.setGroupChatMembers('Core', [remote, reviewer])

    expect(room.data.$botMeta.get().planner.groups).toEqual([])
    expect(room.chat.$groupChats.get().Core.members).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ connectionId: 'remote-1', name: 'planner', sourceScoped: true }),
        expect.objectContaining({ name: 'reviewer' })
      ])
    )
  })
})

describe('setGroupMembership (per-Bot Manage groups)', () => {
  it('unchecking a room writes the stored descriptors too, so the Bot loses its seat (#91329)', async () => {
    const room = await loadRoom()
    seedCore(room)

    await room.members.setGroupMembership(reviewer, 'Core', false)

    expect(room.data.$botMeta.get().reviewer.groups).toEqual([])
    expect(seatedNames(room)).toEqual(['programmer'])
    expect(room.chat.$groupChats.get().Core.members?.map(member => member.name)).toEqual(['programmer'])

    await room.members.setGroupMembership(reviewer, 'Core', true)

    expect(seatedNames(room)).toEqual(['programmer', 'reviewer'])
  })
})
