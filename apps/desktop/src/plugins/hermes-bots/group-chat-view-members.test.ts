import { beforeEach, describe, expect, it, vi } from 'vitest'

import type * as data from './data'
import type * as groupChat from './group-chat'
import type * as members from './group-chat-view-members'
import type * as groupMembership from './group-membership'
import type * as groupRounds from './group-rounds'
import { createGroupGateway, drain, runTimersInline, scriptedStorage } from './group-test-utils'
import type { ScriptedGateway } from './group-test-utils'
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
  gateway: ScriptedGateway
  members: typeof members
  membership: typeof groupMembership
  rounds: typeof groupRounds
}

async function loadRoom(): Promise<Room> {
  vi.resetModules()
  const gateway = createGroupGateway()

  for (const key of Object.keys(host)) {
    delete host[key]
  }

  Object.assign(host, gateway.host)

  const [chat, d, m, membership, rounds, shared] = await Promise.all([
    import('./group-chat'),
    import('./data'),
    import('./group-chat-view-members'),
    import('./group-membership'),
    import('./group-rounds'),
    import('./shared')
  ])

  shared.setPluginCtx(scriptedStorage(gateway.storage))

  return { chat, data: d, gateway, members: m, membership, rounds }
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

  it("does not replay a removed member's held messages as fresh delta after re-adding it", async () => {
    const room = await loadRoom()
    seedCore(room, {
      heldMessages: { reviewer: ['held-before-removal'] },
      log: [
        {
          at: 1,
          from: { kind: 'user', name: 'You' },
          id: 'held-before-removal',
          text: 'STALE_PRE_REMOVAL_TEXT',
          thread: 'old-thread'
        }
      ]
    })

    await room.members.setGroupMembership(reviewer, 'Core', false)
    await room.members.setGroupMembership(reviewer, 'Core', true)
    room.rounds.sendToGroupChat('Core', [reviewer], '@reviewer fresh task')
    await drain(() => Boolean(room.chat.$groupChats.get().Core.running))

    const prompt = room.gateway.calls[0].prompt
    const delta = prompt.split('New messages in the room since your last turn (oldest first):')[1] || ''

    expect(room.chat.$groupChats.get().Core.heldMessages?.reviewer).toBeUndefined()
    expect(prompt).toContain('Historical room context from other threads')
    expect(delta).not.toContain('STALE_PRE_REMOVAL_TEXT')
    expect(delta).toContain('fresh task')
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

describe('a room that spans machines', () => {
  const remote: RosterRow = {
    connectionId: 'remote-1',
    connectionKind: 'remote',
    name: 'scout',
    remoteSource: true,
    route: { connectionId: 'remote-1', mode: 'remote', profile: 'scout', targetProfile: 'scout' },
    sourceScoped: true
  }

  it('stays Desktop-driven after its last remote member is removed; a local-only edit never marks one', async () => {
    const room = await loadRoom()
    seedCore(room, { members: room.membership.durableGroupChatMembers([programmer, reviewer, remote]) })
    room.data.$lastRoster.set([programmer, planner, reviewer, remote])

    await room.members.setGroupChatMembers('Core', [programmer, reviewer])

    expect(room.chat.$groupChats.get().Core.members?.map(member => member.name)).toEqual(['programmer', 'reviewer'])
    expect(room.chat.$groupChats.get().Core.engine).toBe('desktop')

    const local = await loadRoom()
    seedCore(local)
    await local.members.setGroupMembership(reviewer, 'Core', false)

    expect(local.chat.$groupChats.get().Core.engine).toBeUndefined()
  })

  it('a remote member removed in Manage members gets no seat and no turn on the next @all send', async () => {
    const room = await loadRoom()
    seedCore(room, { members: room.membership.durableGroupChatMembers([programmer, reviewer, remote]) })
    room.data.$lastRoster.set([programmer, planner, reviewer, remote])
    // Creation writes groups[] for every seated bot, the remote one included.
    room.data.$botMeta.set({
      ...room.data.$botMeta.get(),
      [room.data.botMetaKey(remote)]: { groups: ['Core'] }
    })
    expect(seatedNames(room)).toEqual(['programmer', 'reviewer', 'scout'])

    await room.members.setGroupChatMembers('Core', [programmer, reviewer])

    expect(seatedNames(room)).toEqual(['programmer', 'reviewer'])

    room.rounds.sendToGroupChat(
      'Core',
      seatedNames(room).map(name => ({ name })),
      '@all go'
    )
    await drain(() => Boolean(room.chat.$groupChats.get().Core.running))

    expect(room.gateway.calls.map(call => call.profile)).not.toContain('scout')
    expect(room.chat.$groupChats.get().Core.members?.map(member => member.name)).toEqual(['programmer', 'reviewer'])
  })

  it('removing a member whose machine does not answer fails and keeps all three seats, through reconnect too', async () => {
    const room = await loadRoom()
    seedCore(room, { members: room.membership.durableGroupChatMembers([programmer, reviewer, remote]) })
    room.data.$lastRoster.set([programmer, planner, reviewer, remote])
    room.data.$botMeta.set({ ...room.data.$botMeta.get(), [room.data.botMetaKey(remote)]: { groups: ['Core'] } })

    // The machine went away after the roster painted it reachable.
    host.requestProfile = async () => {
      throw new Error('connection refused')
    }

    await expect(room.members.setGroupChatMembers('Core', [programmer, reviewer])).rejects.toThrow(/Scout/)

    expect(seatedNames(room)).toEqual(['programmer', 'reviewer', 'scout'])
    expect(room.chat.$groupChats.get().Core.members?.map(member => member.name)).toEqual([
      'programmer',
      'reviewer',
      'scout'
    ])

    // Reconnect: that machine's ui_meta still names the room, and now agrees with this Desktop.
    const { mergeServerMeta } = await import('./profile-ops')
    mergeServerMeta(
      [{ ...remote, sourceReachable: true, ui_meta: { 'hermes-bots': { groups: ['Core'] } } }],
      Date.now() + 120_000
    )

    expect(seatedNames(room)).toEqual(['programmer', 'reviewer', 'scout'])
  })

  it('refuses before writing anything when the removed member is already shown offline', async () => {
    const room = await loadRoom()
    const offline = { ...remote, connectionLabel: 'Homelab', sourceReachable: false }
    seedCore(room, { members: room.membership.durableGroupChatMembers([programmer, reviewer, offline]) })
    room.data.$lastRoster.set([programmer, planner, reviewer, offline])
    room.data.$botMeta.set({ ...room.data.$botMeta.get(), [room.data.botMetaKey(offline)]: { groups: ['Core'] } })
    const before = room.data.$botMeta.get()

    await expect(room.members.setGroupChatMembers('Core', [programmer, planner])).rejects.toThrow(/Homelab/)

    expect(room.data.$botMeta.get()).toEqual(before)
    expect(seatedNames(room)).toEqual(['programmer', 'reviewer', 'scout'])
  })

  it('Manage groups: adding or removing a bot whose machine does not answer fails, and reconnect agrees', async () => {
    const room = await loadRoom()
    seedCore(room, { members: room.membership.durableGroupChatMembers([programmer, reviewer, remote]) })
    room.data.$lastRoster.set([programmer, planner, reviewer, remote])
    room.data.$botMeta.set({ ...room.data.$botMeta.get(), [room.data.botMetaKey(remote)]: { groups: ['Core'] } })
    const { mergeServerMeta } = await import('./profile-ops')

    host.requestProfile = async () => {
      throw new Error('connection refused')
    }

    await expect(room.members.setGroupMembership(remote, 'Core', false)).rejects.toThrow(/Scout/)
    expect(room.data.$botMeta.get()[room.data.botMetaKey(remote)].groups).toEqual(['Core'])
    expect(room.chat.$groupChats.get().Core.members?.map(member => member.name)).toEqual([
      'programmer',
      'reviewer',
      'scout'
    ])
    mergeServerMeta(
      [{ ...remote, sourceReachable: true, ui_meta: { 'hermes-bots': { groups: ['Core'] } } }],
      Date.now() + 120_000
    )
    expect(seatedNames(room)).toEqual(['programmer', 'reviewer', 'scout'])

    // Its machine has no rooms; joining one while it does not answer is refused too.
    room.data.$botMeta.set({ ...room.data.$botMeta.get(), [room.data.botMetaKey(remote)]: { groups: [] } })
    seedCore(room)
    room.data.$lastRoster.set([programmer, planner, reviewer, remote])

    await expect(room.members.setGroupMembership(remote, 'Core', true)).rejects.toThrow(/Scout/)
    expect(room.data.$botMeta.get()[room.data.botMetaKey(remote)].groups).toEqual([])
    expect(room.chat.$groupChats.get().Core.members?.map(member => member.name)).toEqual(['programmer', 'reviewer'])
    mergeServerMeta(
      [{ ...remote, sourceReachable: true, ui_meta: { 'hermes-bots': { groups: [] } } }],
      Date.now() + 240_000
    )
    expect(seatedNames(room)).toEqual(['programmer', 'reviewer'])
  })

  it('Manage groups refuses before writing when the bot is already shown offline', async () => {
    const room = await loadRoom()
    const offline = { ...remote, connectionLabel: 'Homelab', sourceReachable: false }
    seedCore(room)
    room.data.$lastRoster.set([programmer, planner, reviewer, offline])
    const before = room.data.$botMeta.get()

    await expect(room.members.setGroupMembership(offline, 'Core', true)).rejects.toThrow(/Homelab/)

    expect(room.data.$botMeta.get()).toEqual(before)
    expect(seatedNames(room)).toEqual(['programmer', 'reviewer'])
  })
})
