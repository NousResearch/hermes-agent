import { beforeEach, describe, expect, it, vi } from 'vitest'

import type * as data from './data'
import type * as groupChat from './group-chat'
import type * as members from './group-chat-view-members'
import { createGroupGateway, runTimersInline, scriptedStorage } from './group-test-utils'
import type { RosterRow } from './types'

// The room-side member picker's save must leave the room reading the SAME
// roster from its two sources — local groups[] metadata and stored member
// descriptors.

const { host } = vi.hoisted(() => ({ host: {} as Record<string, unknown> }))

vi.mock('@hermes/plugin-sdk', async () => {
  const { pluginSdkMock } = await import('./group-test-utils')

  return pluginSdkMock(host)
})

interface Room {
  chat: typeof groupChat
  data: typeof data
  members: typeof members
}

async function loadRoom(): Promise<Room> {
  vi.resetModules()
  const gateway = createGroupGateway()

  for (const key of Object.keys(host)) {
    delete host[key]
  }

  Object.assign(host, gateway.host)

  const [chat, d, m, shared] = await Promise.all([
    import('./group-chat'),
    import('./data'),
    import('./group-chat-view-members'),
    import('./shared')
  ])

  shared.setPluginCtx(scriptedStorage(gateway.storage))

  return { chat, data: d, members: m }
}

beforeEach(() => {
  runTimersInline()
})

describe('setGroupChatMembers', () => {
  it('keeps a same-named local Bot out when selecting only its Connection counterpart', async () => {
    const room = await loadRoom()
    const local: RosterRow = { name: 'planner', title: 'Local Planner' }
    const reviewer: RosterRow = { name: 'reviewer' }
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
    expect(room.chat.$groupChats.get().Core.members).toEqual(expect.arrayContaining([
      expect.objectContaining({ connectionId: 'remote-1', name: 'planner', sourceScoped: true }),
      expect.objectContaining({ name: 'reviewer' })
    ]))
  })
})
