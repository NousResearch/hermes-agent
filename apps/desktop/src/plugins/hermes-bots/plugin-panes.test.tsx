/**
 * Bot Mode's pane layout contract, asserted by running the real `register()`
 * against a recording plugin context:
 *
 *  - the Bots pane center-stacks into the sessions zone (a SESSIONS | BOTS tab
 *    strip), never splits below it, and carries the ENFORCED dock invariant so
 *    every boot re-homes a stacked install. No heal token, no user-placed
 *    exemption — the retired one-shot heal burned its token even when its
 *    guards skipped the move, so exactly the users who had dragged their panes
 *    stayed stacked forever;
 *  - the Scheduled jobs (internally `routines`) pane only exists while a BOT
 *    CHAT owns the main workspace and the Bots pane is on screen. It is
 *    registered and unregistered through the contribution disposer, driven by
 *    the feature-detected `host.paneVisibility` export, with the
 *    always-registered fallback kept for older desktops. Cron jobs are
 *    bot-scoped, so the tile must not sit beside a group chat.
 */

import type * as HermesSdk from '@hermes/plugin-sdk'
import type { PluginContext } from '@hermes/plugin-sdk'
import { atom } from 'nanostores'
import type { ReactNode } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// The app provider the plugin's tab label renders under; a plugin test may reach it.
// eslint-disable-next-line no-restricted-imports
import { I18nProvider } from '@/i18n'

import type * as DataModule from './data'
import type * as RoutingModule from './routing'

const mocks = vi.hoisted(() => ({
  botChatOwnsWorkspace: vi.fn(() => false),
  paneVisibility: vi.fn(),
  sessionOwnsWorkspace: vi.fn(() => false),
  setWorkspaceScope: vi.fn(),
  undismissPane: vi.fn()
}))

vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const original = await importOriginal<typeof HermesSdk>()

  return {
    ...original,
    host: {
      ...original.host,
      onEvent: undefined,
      paneVisibility: mocks.paneVisibility,
      setWorkspaceScope: mocks.setWorkspaceScope,
      undismissPane: mocks.undismissPane
    }
  }
})

// Everything below is a boundary this test does not exercise: clocks, sockets,
// storage sweeps and the panes' own render trees.
vi.mock('./avatar', () => ({ startFaceClock: vi.fn(), stopFaceClock: vi.fn() }))
vi.mock('./relay', () => ({ startBotRelay: vi.fn(), stopBotRelay: vi.fn() }))
vi.mock('./session-sweep', () => ({ startHideSweepScheduler: vi.fn() }))
vi.mock('./canonical-chat', () => ({ openBotCanonicalChat: vi.fn() }))
vi.mock('./chat-empty', () => ({ BotChatEmpty: () => null }))
vi.mock('./hygiene', () => ({ annotateOrphanedGroupChatMembers: () => ({ changed: false, rooms: {} }) }))
vi.mock('./cron', () => ({ bindProfileSync: () => () => undefined, RoutinesPane: () => null }))
vi.mock('./roster-pane', () => ({
  botChatOwnsWorkspace: mocks.botChatOwnsWorkspace,
  BotsPane: () => null,
  releaseStaleOpenBotChat: vi.fn(),
  selectedRosterBot: () => null,
  sessionOwnsWorkspace: mocks.sessionOwnsWorkspace
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
vi.mock('./data', async importOriginal => {
  const original = await importOriginal<typeof DataModule>()

  return { ...original, migrateBotMeta: async () => undefined }
})
vi.mock('./routing', async importOriginal => {
  const original = await importOriginal<typeof RoutingModule>()

  return { ...original, setBotsWorkspaceOwner: vi.fn() }
})

const plugin = (await import('./plugin')).default

interface Registration {
  area: string
  data?: Record<string, unknown>
  id: string
}

/** A recording `PluginContext`: registrations, their disposers, teardown. */
function recordingContext() {
  const disposers: (() => void)[] = []
  const registrations: Registration[] = []
  const unregisters = new Map<string, () => void>()

  const ctx = {
    i18n: { register: () => () => undefined, t: (key: string) => key },
    onDispose: (fn: () => void) => disposers.push(fn),
    register: (registration: Registration) => {
      registrations.push(registration)

      const unregister = vi.fn(() => {
        registrations.splice(registrations.indexOf(registration), 1)
      })

      unregisters.set(registration.id, unregister)

      return unregister
    },
    storage: { get: async () => undefined, set: async () => undefined }
  }

  return {
    ctx: ctx as unknown as PluginContext,
    dispose: () => disposers.forEach(fn => fn()),
    find: (id: string) => registrations.find(registration => registration.id === id),
    unregisters
  }
}

/** Nanostore stand-ins for the SDK's per-pane visibility stores. */
function paneStores() {
  const stores = new Map<string, ReturnType<typeof atom<boolean>>>()

  mocks.paneVisibility.mockImplementation((id: string) => {
    if (!stores.has(id)) {
      stores.set(id, atom(false))
    }

    return stores.get(id)
  })

  return (id: string) => {
    mocks.paneVisibility(id)

    return stores.get(id)!
  }
}

/** The plugin defers one reconcile to a macrotask so it never re-enters the
 *  tree store mid-mutation. */
const settle = () => new Promise(resolve => setTimeout(resolve, 0))

beforeEach(() => {
  vi.clearAllMocks()
  mocks.botChatOwnsWorkspace.mockReturnValue(false)
  mocks.sessionOwnsWorkspace.mockReturnValue(false)
})

afterEach(() => {
  vi.useRealTimers()
})

describe('the Bots pane dock', () => {
  it('renders its tab label from the live locale, not the register-time string', () => {
    paneStores()

    const harness = recordingContext()

    // Registration runs at module import, before the app has loaded
    // `display.language`: the string `title` is English here no matter what.
    plugin.register(harness.ctx)

    const tabTitle = harness.find('pane')!.data!.tabTitle as () => ReactNode

    const inLocale = (locale: string) =>
      renderToStaticMarkup(
        <I18nProvider configClient={null} initialLocale={locale}>
          {tabTitle()}
        </I18nProvider>
      )

    expect(inLocale('en')).toBeTruthy()
    expect(inLocale('ru')).not.toBe(inLocale('en'))

    harness.dispose()
  })
})

describe('the Scheduled jobs pane', () => {
  it('stays unregistered until a bot chat owns the workspace', async () => {
    const store = paneStores()
    const harness = recordingContext()

    plugin.register(harness.ctx)
    await settle()

    expect(harness.find('routines')).toBeUndefined()

    mocks.botChatOwnsWorkspace.mockReturnValue(true)
    store(`hermes-bots:pane`).set(true)

    expect(harness.find('routines')).toBeTruthy()

    harness.dispose()
  })

  it('unregisters when Bot Mode leaves the screen', async () => {
    const store = paneStores()
    const harness = recordingContext()

    mocks.botChatOwnsWorkspace.mockReturnValue(true)
    plugin.register(harness.ctx)
    await settle()

    expect(harness.find('routines')).toBeTruthy()

    mocks.botChatOwnsWorkspace.mockReturnValue(false)
    store(`hermes-bots:pane`).set(true)
    store(`hermes-bots:pane`).set(false)

    expect(harness.unregisters.get('routines')).toHaveBeenCalled()
    expect(harness.find('routines')).toBeUndefined()

    harness.dispose()
  })

  it('keeps the tile alive while the tile itself holds focus', async () => {
    const store = paneStores()
    const harness = recordingContext()

    mocks.botChatOwnsWorkspace.mockReturnValue(true)
    plugin.register(harness.ctx)
    await settle()

    // Clicking the tile drops bot-chat workspace ownership for a beat. A pane
    // must never unregister itself out from under its own click.
    store(`hermes-bots:routines`).set(true)
    mocks.botChatOwnsWorkspace.mockReturnValue(false)
    store(`hermes-bots:pane`).set(true)

    expect(harness.unregisters.get('routines')).not.toHaveBeenCalled()
    expect(harness.find('routines')).toBeTruthy()

    harness.dispose()
  })

  it('drops a remembered Close only on entering Bot Mode, not on every ownership regain', async () => {
    const store = paneStores()
    const harness = recordingContext()

    mocks.botChatOwnsWorkspace.mockReturnValue(true)
    store(`hermes-bots:pane`).set(true)
    plugin.register(harness.ctx)
    await settle()

    // Boot straight into a bot chat: the pane arrives and a Close from a past
    // launch is dropped once (#102224).
    expect(mocks.undismissPane).toHaveBeenCalledTimes(1)
    expect(mocks.undismissPane).toHaveBeenCalledWith('hermes-bots:routines')

    // The user ✕-es the pane, opens a group room (the tile must not sit
    // beside a group chat) and comes back to the bot chat — all inside one
    // Bots session. Re-registration must not undo their Close.
    const { $groupChatWorkspace } = await import('./group-chat')
    mocks.botChatOwnsWorkspace.mockReturnValue(false)
    $groupChatWorkspace.set({ id: 'room' } as never)
    expect(harness.find('routines')).toBeUndefined()

    mocks.botChatOwnsWorkspace.mockReturnValue(true)
    $groupChatWorkspace.set(null)
    expect(harness.find('routines')).toBeTruthy()
    expect(mocks.undismissPane).toHaveBeenCalledTimes(1)

    // Leaving Bot Mode and coming back is the ask for the bot's chrome again.
    mocks.botChatOwnsWorkspace.mockReturnValue(false)
    store(`hermes-bots:pane`).set(false)
    mocks.botChatOwnsWorkspace.mockReturnValue(true)
    store(`hermes-bots:pane`).set(true)
    expect(mocks.undismissPane).toHaveBeenCalledTimes(2)

    harness.dispose()
  })

  it('stops every lifecycle listener when the plugin is disabled', async () => {
    const store = paneStores()
    const harness = recordingContext()

    plugin.register(harness.ctx)
    await settle()
    harness.dispose()

    // A disable → re-enable cycle used to stack a duplicate listener per cycle.
    mocks.botChatOwnsWorkspace.mockReturnValue(true)
    store(`hermes-bots:pane`).set(true)

    expect(harness.find('routines')).toBeUndefined()
  })
})

describe('a desktop without host.paneVisibility', () => {
  it('keeps the always-registered pane', async () => {
    const { host } = await import('@hermes/plugin-sdk')
    const restore = host.paneVisibility

    // @ts-expect-error modelling an older SDK that lacks the export entirely
    host.paneVisibility = undefined

    const harness = recordingContext()

    plugin.register(harness.ctx)

    expect(harness.find('routines')).toBeTruthy()

    harness.dispose()
    host.paneVisibility = restore
  })
})
