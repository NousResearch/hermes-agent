/**
 * A Desktop-driven room's `engine: 'desktop'` reaches plugin storage through
 * every writer of the stored room map: a room write, the post-merge rewrite and
 * the rewrite of the rooms left after a disband. Hydration reads the field back
 * (group-chat-hydrate-engine.test.ts), so a writer that drops it hands an edited
 * all-local room back to the membership check after a reload.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest'

import type * as groupChat from './group-chat'
import type * as groupChatView from './group-chat-view'
import { createGroupGateway, runTimersInline, scriptedStorage } from './group-test-utils'
import type { ScriptedGateway } from './group-test-utils'
import type { GroupChat } from './types'

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

const stored = (room: Room) => (room.gateway.storage.get('group-chats') || {}) as Record<string, GroupChat>

function seed(room: Room) {
  for (const [name, engine] of [['Mixed', 'desktop'], ['Plain', undefined]] as const) {
    room.chat.updateGroupChat(
      name,
      r => ({ ...r, members: [{ name: 'alpha' }, { name: 'beta' }], roomId: `room-${name}`, ...(engine ? { engine } : {}) }),
      { sync: false }
    )
  }
}

beforeEach(() => {
  runTimersInline()
})

describe('engine reaches plugin storage', () => {
  it('through a room write and the post-merge rewrite', async () => {
    const room = await loadRoom()
    seed(room)

    expect(stored(room).Mixed.engine).toBe('desktop')
    expect(stored(room).Plain.engine).toBeUndefined()

    room.gateway.storage.delete('group-chats')
    await room.chat.persistGroupChatRooms()

    expect(stored(room).Mixed.engine).toBe('desktop')
    expect(stored(room).Plain.engine).toBeUndefined()
  })

  it('through the rewrite of the rooms a disband leaves', async () => {
    const room = await loadRoom()
    seed(room)
    room.chat.updateGroupChat('Gone', r => ({ ...r, members: [] }), { sync: false })

    await room.view.disbandGroupChat('Gone', [])

    expect(Object.keys(stored(room)).sort()).toEqual(['Mixed', 'Plain'])
    expect(stored(room).Mixed.engine).toBe('desktop')
  })
})
