import { beforeEach, describe, expect, it, vi } from 'vitest'

import type * as groupChat from './group-chat'
import type * as groupChatView from './group-chat-view'
import type * as groupPanes from './group-panes'
import { createGroupGateway, runTimersInline, scriptedStorage } from './group-test-utils'
import type { ScriptedGateway } from './group-test-utils'
import type * as rosterActions from './roster-actions'
import type { GroupChat, RosterRow } from './types'

// Clicking a bot while a group room owns the center is a HANDOFF: the room's
// main-window tab has to be retired before the canonical chat opens, or two
// surfaces fight for the center. And if the open fails, the room the user was
// reading has to come back — the failed action must not steal the center.

const { host } = vi.hoisted(() => ({ host: {} as Record<string, unknown> }))

const { openBotCanonicalChat, prepareBotSource } = vi.hoisted(() => ({
  openBotCanonicalChat: vi.fn(),
  prepareBotSource: vi.fn()
}))

vi.mock('@hermes/plugin-sdk', async () => {
  const { pluginSdkMock } = await import('./group-test-utils')
  const base = await pluginSdkMock(host)

  return {
    ...base,
    ackStoredSessionId: vi.fn(),
    haptic: vi.fn(),
    markSessionUnreadFinished: vi.fn()
  }
})

vi.mock('./canonical-chat', () => ({
  CANONICAL_CHAT_TITLE: 'Bot Chat',
  notifyBotOpenFailure: (...args: unknown[]) => {
    failures.push(args)
  },
  openBotCanonicalChat: (...args: unknown[]) => openBotCanonicalChat(...args),
  prepareBotSource: (...args: unknown[]) => prepareBotSource(...args)
}))

const failures: unknown[][] = []

interface Room {
  actions: typeof rosterActions
  chat: typeof groupChat
  gateway: ScriptedGateway
  panes: typeof groupPanes
  view: typeof groupChatView
}

async function loadRoom(): Promise<Room> {
  vi.resetModules()
  const gateway = createGroupGateway()

  for (const key of Object.keys(host)) {
    delete host[key]
  }

  Object.assign(host, gateway.host)

  const [actions, chat, panes, view, shared] = await Promise.all([
    import('./roster-actions'),
    import('./group-chat'),
    import('./group-panes'),
    import('./group-chat-view'),
    import('./shared')
  ])

  shared.setPluginCtx(scriptedStorage(gateway.storage))

  return { actions, chat, gateway, panes, view }
}

const BOT: RosterRow = { name: 'alpha', title: 'Alpha' }

/** Seat a room and front it as a main-window tab, recording tab closes in
 *  the order they happen relative to the canonical open. */
function registerGroup(room: Room, group: string, timeline: string[]) {
  room.chat.$groupChats.set({
    ...room.chat.$groupChats.get(),
    [group]: { log: [], sessions: {}, watermarks: {} }
  } as unknown as Record<string, GroupChat>)
  host.openWorkspace = (id: string) => () => timeline.push(`close:${id}`)
  room.view.openGroupChat(group)
}

beforeEach(() => {
  runTimersInline()
  failures.length = 0
  prepareBotSource.mockReset().mockResolvedValue(undefined)
  openBotCanonicalChat.mockReset().mockResolvedValue({ openedId: 'stored-chat', registryId: 'stored-chat' })
})

describe('opening a bot from a fronted room', () => {
  it('retires the group tab BEFORE the canonical open', async () => {
    const timeline: string[] = []
    const room = await loadRoom()
    registerGroup(room, 'Core', timeline)
    openBotCanonicalChat.mockImplementation(async () => {
      timeline.push('canonicalOpen')

      return { openedId: 'stored-chat', registryId: 'stored-chat' }
    })

    expect(await room.actions.openRosterBot(BOT)).toBe(true)
    expect(timeline.filter(event => event.includes(':group:'))).toHaveLength(1)
    expect(timeline.findIndex(event => event.includes(':group:'))).toBeLessThan(timeline.indexOf('canonicalOpen'))
    expect(room.panes.groupChatMainTabs.has('Core')).toBe(false)
  })

  it('is safe with no room fronted', async () => {
    const timeline: string[] = []
    const room = await loadRoom()
    host.openWorkspace = (id: string) => () => timeline.push(`close:${id}`)

    expect(await room.actions.openRosterBot(BOT)).toBe(true)
    expect(timeline).toHaveLength(0)
  })
})

describe('a failed open must not steal the center', () => {
  it('restores the group and surfaces the error when the canonical open rejects', async () => {
    const timeline: string[] = []
    const room = await loadRoom()
    registerGroup(room, 'Core', timeline)
    openBotCanonicalChat.mockRejectedValue(new Error('canonical open failed'))

    expect(await room.actions.openRosterBot(BOT)).toBe(false)
    expect(timeline.filter(event => event.includes(':group:'))).toHaveLength(1)
    expect(room.chat.$groupChatWorkspace.get()).toBe('Core')
    // Re-fronted, not merely re-selected: the tab is registered again.
    expect(room.panes.groupChatMainTabs.has('Core')).toBe(true)
    expect(failures).toHaveLength(1)
  })

  it('restores the group when source preparation rejects', async () => {
    const room = await loadRoom()
    registerGroup(room, 'Core', [])
    prepareBotSource.mockRejectedValue(new Error('source preparation failed'))

    expect(await room.actions.openRosterBot({ ...BOT, connectionId: 'local', sourceScoped: true })).toBe(false)
    expect(room.chat.$groupChatWorkspace.get()).toBe('Core')
  })

  it('restores the room by its immutable roomId, so a rename mid-open still finds it', async () => {
    const room = await loadRoom()
    room.chat.$groupChats.set({
      Core: { log: [], roomId: 'r-core', sessions: {}, watermarks: {} }
    } as unknown as Record<string, GroupChat>)
    host.openWorkspace = () => () => undefined
    room.view.openGroupChat('Core')
    openBotCanonicalChat.mockImplementation(async () => {
      const rooms = room.chat.$groupChats.get()
      room.chat.$groupChats.set({ Renamed: rooms.Core } as unknown as Record<string, GroupChat>)
      throw new Error('canonical open failed')
    })

    expect(await room.actions.openRosterBot(BOT)).toBe(false)
    expect(room.chat.$groupChatWorkspace.get()).toBe('Renamed')
  })

  it('does not resurrect a room that was disbanded during the open', async () => {
    const room = await loadRoom()
    registerGroup(room, 'Core', [])
    openBotCanonicalChat.mockImplementation(async () => {
      room.chat.$groupChats.set({})
      throw new Error('canonical open failed')
    })

    expect(await room.actions.openRosterBot(BOT)).toBe(false)
    expect(room.chat.$groupChatWorkspace.get()).toBeNull()
  })
})
