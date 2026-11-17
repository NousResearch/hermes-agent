/**
 * A Desktop-driven room's engine (`engine: 'desktop'`, set when the room was
 * created across machines) survives a window reload: the room writer stores it
 * and plugin hydration rebuilds each stored room field by field, so a field
 * dropped on either side hands the room back to the membership check after the
 * next membership edit.
 */

import { describe, expect, it, vi } from 'vitest'

import type * as GroupChatModule from './group-chat'
import type * as I18nModule from './i18n'
import type { RosterRow } from './types'

const { openBotCanonicalChat, onEvent } = vi.hoisted(() => ({
  onEvent: vi.fn((_type: string, listener: (event: { payload?: unknown }) => void) => {
    listeners.push(listener)

    return () => undefined
  }),
  openBotCanonicalChat: vi.fn(async () => ({ openedId: 'bot-chat-tip', registryId: 'bot-chat' }))
}))

const listeners: Array<(event: { payload?: unknown }) => void> = []

vi.mock('@hermes/plugin-sdk', async () => {
  const { atom } = await import('nanostores')

  const stub: unknown = new Proxy(function stubbed() {}, {
    apply: () => stub,
    get: (_target, key) => (key === 'then' ? undefined : stub),
    has: () => true
  })

  const hostMock = {
    onEvent,
    paneVisibility: vi.fn(() => ({ get: () => false, listen: () => () => undefined })),
    request: vi.fn(async () => ({})),
    state: {
      focusedStoredSessionId: { get: () => null, listen: () => () => undefined },
      gateway: { get: () => 'open', listen: () => () => undefined },
      profile: { get: () => 'default', listen: () => () => undefined }
    }
  }

  const known: Record<string, unknown> = { atom, host: hostMock }

  return new Proxy(known, {
    get: (target, key) =>
      typeof key === 'symbol' || key in target ? target[key as string] : key === 'then' ? undefined : stub,
    has: () => true
  })
})

vi.mock('./avatar', () => ({ startFaceClock: vi.fn(), stopFaceClock: vi.fn() }))
vi.mock('./canonical-chat', () => ({
  CANONICAL_CHAT_TITLE: 'Bot Chat',
  isCanonicalChatOnScreen: vi.fn(() => false),
  notifyBotOpenFailure: vi.fn(),
  openBotCanonicalChat
}))
vi.mock('./chat-empty', () => ({ BotChatEmpty: () => null }))
vi.mock('./relay', () => ({ startBotRelay: vi.fn(), stopBotRelay: vi.fn() }))
vi.mock('./session-sweep', () => ({ startHideSweepScheduler: vi.fn() }))
vi.mock('./cron', () => ({ bindProfileSync: () => () => undefined, RoutinesPane: () => null }))
vi.mock('./screen-autoraise', () => ({ startScreenAutoRaise: vi.fn(() => () => undefined) }))
vi.mock('./hygiene', () => ({ annotateOrphanedGroupChatMembers: () => ({ changed: false, rooms: {} }) }))
vi.mock('./group-chat', async () => {
  const { atom: nanoAtom } = await import('nanostores')

  return {
    $groupChats: nanoAtom({}),
    $groupChatWorkspace: nanoAtom(null),
    assignLegacyThreads: (log: unknown[]) => log,
    handleSessionsGatewayTransition: vi.fn(),
    hydrateGroupChatTombstones: vi.fn(async () => undefined),
    pullGroupChatServerState: async () => false,
    scheduleGroupChatServerSync: vi.fn(),
    setGroupChatSyncDisposed: vi.fn(),
    stopGroupChatServerSync: vi.fn(),
    sweepGroupChatMembersForRemovedConnection: vi.fn(),
    updateGroupChat: vi.fn()
  }
})
vi.mock('./user-sections', () => ({ loadBotSections: vi.fn() }))
vi.mock('./data', async () => {
  const { atom: nanoAtom } = await import('nanostores')

  return {
    $botMeta: nanoAtom({}),
    $lastRoster: nanoAtom<RosterRow[]>([]),
    botHandle: (name: string) => name,
    botMentionTag: (name: string) => name,
    botRosterKey: (bot: { connectionId?: string; name?: string } | null | undefined) =>
      `${bot?.connectionId || 'legacy'}::${bot?.name || 'default'}`,
    botSelectionKey: (bot: RosterRow) => bot.name,
    cachedUnionRoster: () => null,
    isActiveRosterBot: () => false,
    migrateBotMeta: vi.fn(async () => undefined),
    primeRoster: vi.fn(async () => undefined),
    resolveRosterMentions: vi.fn(() => ({}))
  }
})
vi.mock('./i18n', async importOriginal => {
  const actual = await importOriginal<typeof I18nModule>()

  return { ...actual, BOTS_LOCALES: {} }
})
const plugin = (await import('./plugin')).default
const { $groupChats } = await import('./group-chat')

// The stored map comes from the real writer, not a hand-built seed: a field the
// writer drops is one hydration never sees.
const stored = await (async () => {
  const writes = new Map<string, unknown>()
  const writer = await vi.importActual<typeof GroupChatModule>('./group-chat')
  const { setPluginCtx } = await import('./shared')
  setPluginCtx({
    storage: { get: async () => null, set: async (key: string, value: unknown) => void writes.set(key, value) }
  } as never)

  writer.updateGroupChat(
    'Mixed',
    room => ({ ...room, engine: 'desktop', members: [{ name: 'alpha' }], roomId: 'r1' }),
    { sync: false }
  )
  writer.updateGroupChat('Plain', room => ({ ...room, members: [{ name: 'beta' }], roomId: 'r2' }), { sync: false })

  return writes.get('group-chats')
})()

describe('group-chat hydration', () => {
  it("keeps a stored room's Desktop engine and adds none to a room without one", async () => {
    try {
      plugin.register({
        i18n: { register: () => () => undefined },
        onDispose: () => undefined,
        register: () => () => undefined,
        storage: {
          get: async (key: string) => (key === 'group-chats' ? stored : undefined),
          set: async () => undefined
        }
      } as never)
    } catch {
      // Registration walks UI surfaces the stub does not model; hydration is
      // started before any throw.
    }

    await vi.waitFor(() => expect(Object.keys($groupChats.get())).toEqual(['Mixed', 'Plain']))

    expect($groupChats.get().Mixed.engine).toBe('desktop')
    expect($groupChats.get().Plain.engine).toBeUndefined()
  })
})
