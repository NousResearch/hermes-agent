/**
 * The three passive guards that ask who owns the center, driven through the
 * real consumers rather than through `activeGroupChat` directly.
 *
 * `group-panes.pane-scope.test.ts` pins the helper. A helper can be right while
 * every consumer still reads the selection atom, which is the state this change
 * exists to leave behind, so these drive the two surfaces that decide what ⌘T
 * does: the sidebar-visibility handler `register()` installs, and
 * `botChatOwnsWorkspace`.
 *
 * The live failure both reproduce: with a room's tab fronted and
 * `$groupChatWorkspace` cleared, the ~5s roster poll republished the workspace
 * scope from the PERSISTED selected bot, and ⌘T created a chat on that bot
 * instead of refusing with "New group conversations start in the group
 * composer."
 */

import type * as HermesSdk from '@hermes/plugin-sdk'
import type { PluginContext } from '@hermes/plugin-sdk'
import { atom } from 'nanostores'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import type * as DataModule from './data'
import type * as RosterPaneModule from './roster-pane'
import type * as RoutingModule from './routing'
import type { RosterRow } from './types'

const BOT: RosterRow = { connectionId: 'local', name: 'scribe' } as RosterRow
const ROOM = 'Alpha, Beta, Gamma'
const BLOCKED = 'New group conversations start in the group composer.'

const mocks = vi.hoisted(() => ({
  paneVisibility: vi.fn(),
  selectedRosterBot: vi.fn(),
  setBotsWorkspaceOwner: vi.fn(),
  setWorkspaceScope: vi.fn()
}))

vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const original = await importOriginal<typeof HermesSdk>()

  return {
    ...original,
    host: {
      ...original.host,
      onEvent: undefined,
      paneVisibility: mocks.paneVisibility,
      setWorkspaceScope: mocks.setWorkspaceScope
    }
  }
})

// Boundaries this test does not exercise: clocks, sockets, storage sweeps,
// render trees. `./group-panes` stays REAL — it is the subject.
vi.mock('./avatar', () => ({ startFaceClock: vi.fn(), stopFaceClock: vi.fn() }))
vi.mock('./relay', () => ({ startBotRelay: vi.fn(), stopBotRelay: vi.fn() }))
vi.mock('./session-sweep', () => ({ startHideSweepScheduler: vi.fn() }))
vi.mock('./canonical-chat', () => ({ openBotCanonicalChat: vi.fn() }))
vi.mock('./chat-empty', () => ({ BotChatEmpty: () => null }))
vi.mock('./hygiene', () => ({ annotateOrphanedGroupChatMembers: () => ({ changed: false, rooms: {} }) }))
vi.mock('./cron', () => ({ bindProfileSync: () => () => undefined, RoutinesPane: () => null }))
vi.mock('./roster-pane', async importOriginal => {
  const original = await importOriginal<typeof RosterPaneModule>()

  return { ...original, BotsPane: () => null, selectedRosterBot: mocks.selectedRosterBot }
})
vi.mock('./group-chat', async () => {
  const { atom: nanoAtom } = await import('nanostores')

  return {
    $groupChats: nanoAtom({}),
    $groupChatWorkspace: nanoAtom(null),
    assignLegacyThreads: (log: unknown[]) => log,
    // Same shape as the real one: group-panes imports the binding, and a
    // one-argument stand-in would hand a future caller the wrong key.
    groupChatRoomKey: (name: string, room: { roomId?: string }) =>
      typeof room?.roomId === 'string' && room.roomId ? `id:${room.roomId}` : `name:${String(name)}`,
    handleSessionsGatewayTransition: vi.fn(),
    pullGroupChatServerState: async () => false,
    scheduleGroupChatServerSync: vi.fn(),
    setGroupChatSyncDisposed: vi.fn(),
    stopGroupChatServerSync: vi.fn(),
    sweepGroupChatMembersForRemovedConnection: vi.fn(),
    updateGroupChat: vi.fn()
  }
})
vi.mock('./data', async importOriginal => {
  const original = await importOriginal<typeof DataModule>()

  return { ...original, migrateBotMeta: async () => undefined }
})
vi.mock('./routing', async importOriginal => {
  const original = await importOriginal<typeof RoutingModule>()

  return { ...original, setBotsWorkspaceOwner: mocks.setBotsWorkspaceOwner }
})

const plugin = (await import('./plugin')).default
const panes = await import('./group-panes')
const { $groupChatWorkspace } = await import('./group-chat')
const { $botsPaneVisible, $openBotChat } = await import('./bot-state')
const { botChatOwnsWorkspace } = await import('./roster-pane')
const { groupWorkspaceOwnerKey } = await import('./group-membership')

/** Nanostore stand-ins for the SDK's per-pane visibility stores, with one room
 *  optionally fronted.
 *
 *  The prefix is applied here rather than taken from `groupChatPaneId`, so a
 *  regression to the bare key cannot make fixture and lookup agree on the wrong
 *  string while these tests stay green. */
function paneStores() {
  const stores = new Map<string, ReturnType<typeof atom<boolean>>>()

  mocks.paneVisibility.mockImplementation((id: string) => {
    if (!stores.has(id)) {
      stores.set(id, atom(false))
    }

    return stores.get(id)
  })

  return {
    front: (group: null | string) => {
      for (const store of stores.values()) {
        store.set(false)
      }

      if (group !== null) {
        const id = `plugin-workspace:${panes.groupChatWorkspaceKey(group)}`

        if (!stores.has(id)) {
          stores.set(id, atom(false))
        }

        stores.get(id)!.set(true)
      }
    },
    store: (id: string) => {
      if (!stores.has(id)) {
        stores.set(id, atom(false))
      }

      return stores.get(id)!
    }
  }
}

function recordingContext() {
  const disposers: (() => void)[] = []

  const ctx = {
    i18n: { register: () => () => undefined, t: (key: string) => key },
    onDispose: (fn: () => void) => disposers.push(fn),
    register: () => vi.fn(),
    storage: { get: async () => undefined, set: async () => undefined }
  }

  return { ctx: ctx as unknown as PluginContext, dispose: () => disposers.forEach(fn => fn()) }
}

beforeEach(() => {
  vi.clearAllMocks()
  mocks.selectedRosterBot.mockReturnValue(BOT)
  $groupChatWorkspace.set(null)
  $botsPaneVisible.set(false)
  $openBotChat.set(null)

  for (const group of [...panes.groupChatMainTabs.keys()]) {
    panes.dropGroupMainTab(group)
  }
})

describe('the sidebar visibility handler', () => {
  it('blocks the new-session target for a fronted room whose selection atom is null', () => {
    const stores = paneStores()
    const harness = recordingContext()

    plugin.register(harness.ctx)

    panes.recordGroupMainTab(ROOM, () => {})
    stores.front(ROOM)
    // The live state: a persisted bot selection still resolves, and the room
    // owns the center anyway.
    $groupChatWorkspace.set(null)

    stores.store('hermes-bots:pane').set(true)

    expect(mocks.setBotsWorkspaceOwner).toHaveBeenCalledWith(groupWorkspaceOwnerKey(ROOM), null, BLOCKED)

    for (const call of mocks.setBotsWorkspaceOwner.mock.calls) {
      expect(call[1]).not.toBe(BOT)
    }

    harness.dispose()
  })

  it('scopes to the selected bot when the room is merely open behind it', () => {
    const stores = paneStores()
    const harness = recordingContext()

    plugin.register(harness.ctx)

    panes.recordGroupMainTab(ROOM, () => {})
    stores.front(null)
    $groupChatWorkspace.set(null)

    stores.store('hermes-bots:pane').set(true)

    expect(mocks.setBotsWorkspaceOwner).toHaveBeenCalledWith(expect.any(String), BOT)
    expect(mocks.setBotsWorkspaceOwner).not.toHaveBeenCalledWith(expect.any(String), null, BLOCKED)

    harness.dispose()
  })

  it('keeps blocking across repeated polls while the room stays fronted', () => {
    const stores = paneStores()
    const harness = recordingContext()

    plugin.register(harness.ctx)

    panes.recordGroupMainTab(ROOM, () => {})
    stores.front(ROOM)
    $groupChatWorkspace.set(null)

    // Each hide/show is one poll cycle republishing the scope. The bug was that
    // the second and every later one landed on the persisted bot.
    for (let poll = 0; poll < 3; poll += 1) {
      stores.store('hermes-bots:pane').set(false)
      stores.store('hermes-bots:pane').set(true)
    }

    expect(mocks.setBotsWorkspaceOwner).toHaveBeenCalledTimes(3)

    for (const call of mocks.setBotsWorkspaceOwner.mock.calls) {
      expect(call).toEqual([groupWorkspaceOwnerKey(ROOM), null, BLOCKED])
    }

    harness.dispose()
  })
})

describe('botChatOwnsWorkspace', () => {
  it('yields the center to a fronted room when the selection atom is null', () => {
    const stores = paneStores()

    panes.recordGroupMainTab(ROOM, () => {})
    stores.front(ROOM)
    $groupChatWorkspace.set(null)
    $botsPaneVisible.set(true)
    $openBotChat.set({ key: 'scribe', openedRegistryId: 'reg-1' })

    // Cronjobs are bot-scoped; seating that tile beside a group room is the
    // visible symptom of this guard reading the wrong owner.
    expect(botChatOwnsWorkspace()).toBe(false)
  })

  it('keeps the center when the room is open but backgrounded', () => {
    const stores = paneStores()

    panes.recordGroupMainTab(ROOM, () => {})
    stores.front(null)
    $groupChatWorkspace.set(null)
    $botsPaneVisible.set(true)
    $openBotChat.set({ key: 'scribe', openedRegistryId: 'reg-1' })

    expect(botChatOwnsWorkspace()).toBe(true)
  })

  // The invariant the helper's own doc understates: selection WINS whenever it
  // is set, so a stale non-null selection blocks the bot scope even with every
  // room pane backgrounded. That is deliberate — an explicit room selection is
  // a real gesture — but it is the case a reader would guess wrong.
  it('still yields to a stale selection whose pane is backgrounded', () => {
    const stores = paneStores()

    panes.recordGroupMainTab(ROOM, () => {})
    stores.front(null)
    $groupChatWorkspace.set(ROOM)
    $botsPaneVisible.set(true)
    $openBotChat.set({ key: 'scribe', openedRegistryId: 'reg-1' })

    expect(panes.frontedGroupChat()).toBeNull()
    expect(panes.activeGroupChat()).toBe(ROOM)
    expect(botChatOwnsWorkspace()).toBe(false)
  })
})
