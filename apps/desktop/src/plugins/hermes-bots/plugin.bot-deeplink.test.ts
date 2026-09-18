/**
 * Inbound `hermes://bot/<profile>` deep links: the OS handler parks the bot
 * name on the shared pending atom (core's deep-link listener), and the plugin
 * claims it — immediately at register for a link that raced plugin startup,
 * and on every later atom change — and opens the bot's canonical Bot Chat
 * addressed by NAME. A hidden canonical chat never appears in the session
 * rows, so the generic session-route path would treat the link as stale and
 * silently drop it (#115134).
 */

import type * as HermesSdk from '@hermes/plugin-sdk'
import type { PluginContext } from '@hermes/plugin-sdk'
import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// Test harness drives the shared pending-link atom directly, as core's
// deep-link listener does.
// eslint-disable-next-line no-restricted-imports
import { $pendingDeepLinkBot } from '@/store/bot-deeplink-open'

const mocks = vi.hoisted(() => ({ openRosterBotByName: vi.fn(async () => true) }))

vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const original = await importOriginal<typeof HermesSdk>()

  return {
    ...original,
    host: {
      ...original.host,
      onEvent: undefined,
      paneVisibility: (id: string) => paneVisibilityStore(id),
      setWorkspaceScope: vi.fn(),
      undismissPane: vi.fn()
    }
  }
})

/** Nanostore stand-ins for the SDK's per-pane visibility stores. */
const visibilityStores = new Map<string, ReturnType<typeof atom>>()

function paneVisibilityStore(id: string) {
  let store = visibilityStores.get(id)

  if (!store) {
    store = atom(false)
    visibilityStores.set(id, store)
  }

  return store
}

vi.mock('./avatar', () => ({ startFaceClock: vi.fn(), stopFaceClock: vi.fn() }))
vi.mock('./relay', () => ({ startBotRelay: vi.fn(), stopBotRelay: vi.fn() }))
vi.mock('./session-sweep', () => ({ startHideSweepScheduler: vi.fn() }))
vi.mock('./canonical-chat', () => ({ openBotCanonicalChat: vi.fn() }))
vi.mock('./chat-empty', () => ({ BotChatEmpty: () => null }))
vi.mock('./hygiene', () => ({ annotateOrphanedGroupChatMembers: () => ({ changed: false, rooms: {} }) }))
vi.mock('./cron', () => ({ bindProfileSync: () => () => undefined, RoutinesPane: () => null }))
vi.mock('./roster-pane', () => ({
  botChatOwnsWorkspace: vi.fn(() => false),
  BotsPane: () => null,
  openRosterBotByName: mocks.openRosterBotByName,
  releaseStaleOpenBotChat: vi.fn(),
  selectedRosterBot: () => null,
  sessionOwnsWorkspace: vi.fn(() => false)
}))
vi.mock('./group-chat', async () => {
  const { atom: nanoAtom } = await import('nanostores')

  return {
    $groupChats: nanoAtom({}),
    $groupChatWorkspace: nanoAtom(null),
    assignLegacyThreads: (log: unknown[]) => log,
    handleSessionsGatewayTransition: vi.fn(),
    pullGroupChatServerState: async () => false,
    scheduleGroupChatServerSync: vi.fn(),
    setGroupChatSyncDisposed: vi.fn(),
    stopGroupChatServerSync: vi.fn(),
    sweepGroupChatMembersForRemovedConnection: vi.fn(),
    updateGroupChat: vi.fn()
  }
})

const plugin = (await import('./plugin')).default

function recordingContext() {
  const disposers: (() => void)[] = []

  const ctx = {
    i18n: { register: () => () => undefined, t: (key: string) => key },
    onDispose: (fn: () => void) => disposers.push(fn),
    register: () => () => undefined,
    storage: { get: async () => undefined, set: async () => undefined }
  }

  return { ctx: ctx as unknown as PluginContext, dispose: () => disposers.forEach(fn => fn()) }
}

beforeEach(() => {
  vi.clearAllMocks()
  $pendingDeepLinkBot.set(null)
})

afterEach(() => {
  $pendingDeepLinkBot.set(null)
})

describe('hermes://bot/<profile> deep links', () => {
  it('opens the named bot canonical chat when the link arrives after register', async () => {
    const harness = recordingContext()
    plugin.register(harness.ctx)
    harness.dispose()

    // A fresh registration listens; the link arrives afterwards.
    const live = recordingContext()
    plugin.register(live.ctx)

    $pendingDeepLinkBot.set('ops')
    await vi.waitFor(() => expect(mocks.openRosterBotByName).toHaveBeenCalledWith('ops'))
    expect($pendingDeepLinkBot.get()).toBe(null)

    live.dispose()
  })

  it('claims a link that raced plugin startup', async () => {
    $pendingDeepLinkBot.set('late-loader')

    const harness = recordingContext()
    plugin.register(harness.ctx)

    expect(mocks.openRosterBotByName).toHaveBeenCalledWith('late-loader')
    expect($pendingDeepLinkBot.get()).toBe(null)

    harness.dispose()
  })

  it('does not open anything without a link', async () => {
    const harness = recordingContext()
    plugin.register(harness.ctx)
    harness.dispose()

    expect(mocks.openRosterBotByName).not.toHaveBeenCalled()
  })

  it('unbinds the listener on dispose', async () => {
    const harness = recordingContext()
    plugin.register(harness.ctx)
    harness.dispose()

    $pendingDeepLinkBot.set('ops')
    await new Promise(resolve => setTimeout(resolve, 0))

    expect(mocks.openRosterBotByName).not.toHaveBeenCalled()
  })
})
