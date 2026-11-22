import { beforeEach, describe, expect, it, vi } from 'vitest'

import type * as groupChat from './group-chat'
import type * as groupChatView from './group-chat-view'
import { createGroupGateway, HELD_GROUP_ROOM, runTimersInline, scriptedStorage } from './group-test-utils'
import type { ScriptedGateway } from './group-test-utils'
import type { DurableGroupChat } from './types'

const { host } = vi.hoisted(() => ({ host: {} as Record<string, unknown> }))

vi.mock('@hermes/plugin-sdk', async () => {
  const { pluginSdkMock } = await import('./group-test-utils')

  return pluginSdkMock(host)
})

interface Room {
  chat: typeof groupChat
  gateway: ScriptedGateway
  view: typeof groupChatView
}

async function loadRoom(): Promise<Room> {
  vi.resetModules()
  const gateway = createGroupGateway()

  for (const key of Object.keys(host)) {
    delete host[key]
  }

  Object.assign(host, gateway.host)

  const [chat, view, shared] = await Promise.all([import('./group-chat'), import('./group-chat-view'), import('./shared')])
  shared.setPluginCtx(scriptedStorage(gateway.storage))

  return { chat, gateway, view }
}

const stored = (room: Room) => (room.gateway.storage.get('group-chats') || {}) as Record<string, DurableGroupChat>

function seed(room: Room) {
  for (const [name, engine] of [['Mixed', 'desktop'], ['Plain', undefined]] as const) {
    room.chat.updateGroupChat(
      name,
      r => ({ ...r, members: [{ name: 'alpha' }, { name: 'beta' }], roomId: `room-${name}`, ...(engine ? { engine } : {}) }),
      { sync: false }
    )
  }

  room.chat.updateGroupChat('Held', r => ({ ...r, ...structuredClone(HELD_GROUP_ROOM) }), { sync: false })
}

beforeEach(() => {
  runTimersInline()
})

describe('the durable room record reaches plugin storage', () => {
  it('through a room write and the post-merge rewrite', async () => {
    const room = await loadRoom()
    seed(room)

    expect(stored(room).Mixed.engine).toBe('desktop')
    expect(stored(room).Plain.engine).toBeUndefined()
    expect(stored(room).Held).toEqual(HELD_GROUP_ROOM)

    room.gateway.storage.delete('group-chats')
    await room.chat.persistGroupChatRooms()

    expect(stored(room).Mixed.engine).toBe('desktop')
    expect(stored(room).Plain.engine).toBeUndefined()
    expect(stored(room).Held).toEqual(HELD_GROUP_ROOM)
  })

  it('through the rewrite after a gateway sync pull', async () => {
    const room = await loadRoom()
    seed(room)
    room.gateway.uiMeta['hermes-bots-groups'] = room.chat.groupChatSyncSnapshot()
    room.gateway.storage.delete('group-chats')

    expect(await room.chat.pullGroupChatServerState('')).toBe(true)

    const { members, ...rest } = stored(room).Held
    const { members: heldMembers, ...heldRest } = HELD_GROUP_ROOM
    expect(rest).toEqual(heldRest)
    expect(members).toEqual(heldMembers?.map(member => ({ ...member, remoteSource: true })))
  })

  it('through the rewrite of the rooms a disband leaves', async () => {
    const room = await loadRoom()
    seed(room)
    room.chat.updateGroupChat('Gone', r => ({ ...r, members: [] }), { sync: false })

    await room.view.disbandGroupChat('Gone', [])

    expect(Object.keys(stored(room)).sort()).toEqual(['Held', 'Mixed', 'Plain'])
    expect(stored(room).Mixed.engine).toBe('desktop')
    expect(stored(room).Held).toEqual(HELD_GROUP_ROOM)
  })

  it("without the tombstone another room's running disband left", async () => {
    const room = await loadRoom()
    seed(room)
    room.chat.updateGroupChat('Busy', r => ({ ...r, members: [], running: true }), { sync: false })
    room.chat.updateGroupChat('Gone', r => ({ ...r, members: [] }), { sync: false })

    await room.view.disbandGroupChat('Busy', [])
    expect(room.chat.$groupChats.get().Busy?.tombstone).toBe(true)
    await room.view.disbandGroupChat('Gone', [])

    expect(Object.keys(stored(room)).sort()).toEqual(['Held', 'Mixed', 'Plain'])
  })
})
