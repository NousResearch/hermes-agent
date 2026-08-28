/**
 * Bot Mode's pane layout contract, asserted by running the real `register()`
 * against a recording plugin context:
 *
 *  - the Bots pane center-stacks into the sessions zone (a SESSIONS | BOTS tab
 *    strip), never splits below it, and carries the ENFORCED dock invariant so
 *    a stacked install with no user-placed record re-homes at boot. No heal
 *    token — the retired one-shot heal burned its token even when its guards
 *    skipped the move, pinning the broken stacked shape forever. User-placed
 *    exemption lives in the tree store (`enforceDockedPanes`), not on the hint;
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
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as DataModule from './data'
import type * as RoutingModule from './routing'

const mocks = vi.hoisted(() => ({
  botChatOwnsWorkspace: vi.fn(() => false),
  paneVisibility: vi.fn(),
  sessionOwnsWorkspace: vi.fn(() => false),
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
  it('center-stacks into the sessions zone as a standing invariant', () => {
    paneStores()

    const harness = recordingContext()

    plugin.register(harness.ctx)

    const data = harness.find('pane')!.data!

    expect(data.dock).toEqual({ enforce: true, pane: 'sessions', pos: 'center' })
    // A 'bottom' split was the old workaround for the lone-pane auto-hide trap.
    expect((data.dock as { pos: string }).pos).not.toBe('bottom')
    // No heal token: the invariant runs at every adoption, unconditionally.
    expect(data).not.toHaveProperty('heal')

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

    const routines = harness.find('routines')!

    expect(routines.data).toMatchObject({
      // Repairs persisted layouts that stranded the tile in the Bots tab strip.
      dock: { enforce: true, pane: 'workspace', pos: 'right' },
      placement: 'main'
    })
    // Glanceable, not something you sit in: it arrives as the right edge's
    // vertical tab and takes no width off the chat until the user opens it.
    expect(routines.data!.defaultCollapsed).toBe(true)

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
