/**
 * The `session.reclaimed` listener's re-resume is a BACKGROUND refresh —
 * issue 121874.
 *
 * When the gateway reaps the runtime behind the OPEN bot chat, the plugin
 * re-resolves the canonical chat so the user's next send doesn't eat the
 * stale-id error. That re-resume used to be a full navigating open: with the
 * user reading the Kanban board (or any other route), the route flipped to
 * the Bot Chat, hijacking the foreground. "Offer; don't hijack"
 * (apps/desktop/AGENTS.md): a background event must never navigate.
 *
 * This pins the listener's contract through the real `plugin.register()`:
 *   - a reclaim for the open claim re-resolves with `background: true` (the
 *     refreshInPlace open — no navigation, no tile minting);
 *   - a reclaim for some other session leaves the claim alone;
 *   - the claim's identities update when the re-resolve reports new ones.
 */

import { beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

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
    $groupClarify: nanoAtom({}),
    $groupNeedsYou: nanoAtom({}),
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

const { $openBotChat, $selectedRosterKey } = await import('./bot-state')
const { $lastRoster } = await import('./data')

const BOT: RosterRow = { connectionId: 'local', name: 'alpha' } as RosterRow

/** Register the real plugin; the stubbed SDK records the reclaim listener. */
function register() {
  listeners.length = 0

  try {
    plugin.register({
      i18n: { register: () => () => undefined },
      onDispose: () => undefined,
      register: () => () => undefined,
      storage: { get: async () => undefined, set: async () => undefined }
    } as never)
  } catch {
    // Registration walks UI surfaces the stub does not model; the listener
    // registered before any throw is what these tests drive.
  }
}

const reclaim = (storedSessionId: string) => ({
  payload: { session_id: 'runtime-x', stored_session_id: storedSessionId },
  type: 'session.reclaimed'
})

beforeAll(async () => {
  await import('./plugin')
}, 120_000)

beforeEach(() => {
  vi.clearAllMocks()
  openBotCanonicalChat.mockClear()
  openBotCanonicalChat.mockResolvedValue({ openedId: 'bot-chat-tip', registryId: 'bot-chat' })
  listeners.length = 0
  $openBotChat.set(null)
  $selectedRosterKey.set('local::alpha')
  $lastRoster.set([BOT])
  register()
})

describe('session.reclaimed re-resumes the open Bot Chat WITHOUT navigating (issue 121874)', () => {
  it('re-resumes the claimed chat as a background refresh', async () => {
    $openBotChat.set({ key: 'local::alpha', openedRegistryId: 'bot-chat', openedSessionId: 'bot-chat-tip' })

    listeners.forEach(listener => listener(reclaim('bot-chat')))

    expect(openBotCanonicalChat).toHaveBeenCalledTimes(1)
    // THE contract: the re-resume is background — refreshInPlace, never a
    // navigating open that would replace /kanban or any other route.
    expect(openBotCanonicalChat).toHaveBeenCalledWith(BOT, { background: true })
  })

  it('ignores a reclaim for a session the claim does not own', async () => {
    $openBotChat.set({ key: 'local::alpha', openedRegistryId: 'bot-chat', openedSessionId: 'bot-chat-tip' })

    listeners.forEach(listener => listener(reclaim('someone-elses-session')))

    expect(openBotCanonicalChat).not.toHaveBeenCalled()
  })

  it('keeps the claim current when the re-resume reports rotated identities', async () => {
    $openBotChat.set({ key: 'local::alpha', openedRegistryId: 'bot-chat', openedSessionId: 'bot-chat-tip' })
    openBotCanonicalChat.mockResolvedValueOnce({ openedId: 'tip-2', registryId: 'reg-2' })

    listeners.forEach(listener => listener(reclaim('bot-chat')))
    await Promise.resolve()

    expect($openBotChat.get()).toEqual({
      key: 'local::alpha',
      openedRegistryId: 'reg-2',
      openedSessionId: 'tip-2'
    })
  })
})
