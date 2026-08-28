import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

// Bot Mode's main workspace always has ONE clear owner: a bot chat, a group
// chat, or the Bots home. The home exists so the Bots tab never falls through
// to the ownerless Sessions composer.
//
// The invariant under test is #90149's: an existing resource carries its exact
// owner. Selecting or RESTORING a bot is presentation only — it must never
// activate a gateway, open a chat, create a session, or route a remote bot
// through whatever connection happens to be live.

const pluginSource = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

function load({ focusedStoredSessionId = null, paneVisibility = true, openWorkspace = true } = {}) {
  const values = new Map()
  const atom = initial => {
    const slot = {
      get: () => values.get(slot),
      set: value => {
        values.set(slot, value)
        for (const fn of slot.__listeners || []) {
          fn(value)
        }
      },
      listen: fn => {
        slot.__listeners = [...(slot.__listeners || []), fn]
        return () => {
          slot.__listeners = (slot.__listeners || []).filter(entry => entry !== fn)
        }
      }
    }
    values.set(slot, initial)
    return slot
  }

  const opened = []
  const closed = []
  const notifications = []
  const requests = []
  const invalidations = []
  const sessionOpens = []
  const workspaceScopes = []
  const paneVisible = new Map()
  const focused = atom(focusedStoredSessionId)

  const host = {
    state: {
      profile: { get: () => 'default', listen: () => undefined },
      gateway: { get: () => 'open', listen: () => undefined },
      focusedStoredSessionId: focused
    },
    request: (method, params) => {
      requests.push({ method, params })
      return Promise.resolve({})
    },
    requestProfile: (route, method, params) => {
      requests.push({ method, params, route })
      if (method === 'profiles.list') {
        return Promise.resolve({ profiles: [{ name: route.targetProfile || route.profile }] })
      }
      if (method === 'session.list') {
        return Promise.resolve({
          sessions: [{ id: `chat-${route.connectionId}-${route.profile}`, title: 'Bot Chat', message_count: 1 }]
        })
      }
      return Promise.resolve({})
    },
    openSession: async (id, options) => sessionOpens.push({ id, options }),
    setWorkspaceScope: (mode, ownerKey = null) => {
      workspaceScopes.push({ mode, ownerKey })
      return true
    },
    notify: params => notifications.push(params),
    notifyError: (error, fallback) => notifications.push({ kind: 'error', message: fallback, error }),
    ensureAgent: async () => undefined,
    activeConnectionId: () => 'local'
  }

  if (openWorkspace) {
    host.openWorkspace = (id, options) => {
      const entry = { id, options, disposed: false }
      opened.push(entry)
      paneVisible.set(`plugin-workspace:${id}`, true)

      return () => {
        entry.disposed = true
        closed.push(entry)
        paneVisible.set(`plugin-workspace:${id}`, false)
        options.onClose?.()
      }
    }
  }

  if (paneVisibility) {
    host.paneVisibility = id => ({
      get: () => paneVisible.get(id) ?? false,
      listen: () => () => undefined
    })
  }

  const context = {
    atom,
    haptic: () => undefined,
    PALETTE_AREA: 'palette',
    COMPOSER_AREAS: { middleware: 'middleware' },
    document: { getElementById: () => null, createElement: () => ({}), head: { appendChild: () => undefined } },
    host,
    queryClient: { invalidateQueries: params => invalidations.push(params) },
    navigator: { clipboard: { writeText: async () => undefined } },
    sdk: new Proxy({}, { get: () => undefined })
  }

  const source = pluginSource
    .replace(/^import\s+\*\s+as\s+sdk\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^import\s+\{[\s\S]*?\}\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^const \{ McpTab, ToolsetConfigPanel \} = sdk\r?\n/m, '')
    .replace(/^import .* from 'react'\r?\n/m, '')
    .replace(/^import .* from 'react\/jsx-runtime'\r?\n/m, '')
    .replace('export default {', 'globalThis.plugin = {')
    .concat(`
globalThis.__home = {
  botRosterKey,
  parseRosterKey,
  ghostRosterOwner,
  rosterWithSelectedOwner,
  reconcileRosterSelection,
  saveSelectedRosterBot,
  clearSelectedRosterBot,
  clearSelectedRosterKey,
  releaseStaleOpenBotChat,
  syncBotsHomeWorkspace,
  openBotsHomeWorkspace,
  closeBotsHomeWorkspace,
  botsHomeMayOpen,
  botsHomeVisible,
  botChatOwnsWorkspace,
  sessionOwnsWorkspace,
  openRosterBot,
  openGroupChat,
  closeGroupChatMainTab,
  prepareBotSource,
  $botMeta,
  $botsPaneVisible,
  $botChatFocused,
  $botsHomeFronted,
  $groupChatWorkspace,
  $groupChats,
  $lastRoster,
  $lastSources,
  $openBotChat,
  $rosterHydrated,
  $selectedBot,
  $selectedRosterHydrated,
  $selectedRosterKey,
  setPluginCtx: value => { pluginCtx = value }
};
`)

  vm.runInNewContext(source, context, { filename: 'plugin.js' })

  return {
    ...context.__home,
    closed,
    focused,
    host,
    invalidations,
    notifications,
    opened,
    paneVisible,
    requests,
    sessionOpens,
    workspaceScopes
  }
}

/** Every door that would create, activate, or route something. A passive
 *  selection must touch none of them. */
function assertNothingRouted(t, label) {
  assert.deepEqual(t.requests, [], `${label}: no gateway RPC`)
  assert.deepEqual(t.opened.filter(entry => entry.id !== 'hermes-bots:home'), [], `${label}: no chat surface opened`)
}

// ── selection is source-qualified and presentation-only ─────────────────────

test('selection persists the source-qualified key, not the bare profile name', () => {
  const t = load()
  const writes = []
  t.setPluginCtx({ storage: { set: (key, value) => writes.push({ key, value }) } })

  t.saveSelectedRosterBot({ connectionId: 'work-vps', name: 'researcher', remoteSource: true })

  assert.equal(t.$selectedRosterKey.get(), 'work-vps::researcher')
  assert.deepEqual(writes.at(-1), { key: 'selected-roster-bot-v1', value: 'work-vps::researcher' })
  assertNothingRouted(t, 'saving a selection')
})

test('a remote selection keeps the exact-owner selection distinct from a local twin', () => {
  const t = load()
  t.setPluginCtx({ storage: { set: () => undefined } })

  // Same profile name on two gateways — the exact-owner class from #90149.
  t.saveSelectedRosterBot({ connectionId: 'local', name: 'default' })
  assert.equal(t.$selectedBot.get(), 'default')

  t.saveSelectedRosterBot({ connectionId: 'mac-mini', name: 'default', remoteSource: true })

  assert.equal(t.$selectedRosterKey.get(), 'mac-mini::default')
  assert.equal(
    t.$selectedBot.get(),
    'mac-mini::default',
    'the shared selection follows the same exact owner as the roster workspace'
  )
})

test('clearing only fires for the exact selected owner', () => {
  const t = load()
  t.setPluginCtx({ storage: { set: () => undefined } })
  t.saveSelectedRosterBot({ connectionId: 'local', name: 'writer' })

  t.clearSelectedRosterBot({ connectionId: 'mac-mini', name: 'writer' })
  assert.equal(t.$selectedRosterKey.get(), 'local::writer', 'a same-named bot elsewhere must not clear this one')

  t.clearSelectedRosterBot({ connectionId: 'local', name: 'writer' })
  assert.equal(t.$selectedRosterKey.get(), '')
})

test('roster keys round-trip through parseRosterKey', () => {
  const t = load()
  const parsed = key => ({ ...t.parseRosterKey(key) })

  assert.deepEqual(parsed('work-vps::researcher'), { connectionId: 'work-vps', name: 'researcher' })
  assert.deepEqual(parsed('local::default'), { connectionId: 'local', name: 'default' })
  assert.deepEqual(parsed(''), { connectionId: '', name: '' })
  // A key always round-trips from the identity that produced it.
  const bot = { connectionId: 'work-vps', name: 'researcher' }
  assert.deepEqual(parsed(t.botRosterKey(bot)), { connectionId: 'work-vps', name: 'researcher' })
})

// ── hydration ───────────────────────────────────────────────────────────────

test('hydration restores the stored key and flips the hydrated flag', async () => {
  const t = load()
  t.$selectedRosterHydrated.set(false)

  const ctx = {
    storage: { get: key => (key === 'selected-roster-bot-v1' ? 'work-vps::researcher' : undefined), set: () => undefined },
    register: () => () => undefined,
    onDispose: () => undefined
  }

  // Only the selection hydrate is under test; run it the way register() does.
  await Promise.resolve(ctx.storage.get('selected-roster-bot-v1')).then(value => {
    if (typeof value === 'string' && value.trim()) {
      t.$selectedRosterKey.set(value.trim())
    }
  })
  t.$selectedRosterHydrated.set(true)

  assert.equal(t.$selectedRosterKey.get(), 'work-vps::researcher')
  assert.equal(t.$selectedRosterHydrated.get(), true)
  assertNothingRouted(t, 'restoring a selection')
})

test('source contract: every hydrate settle path flips the flag, and none of them opens anything', () => {
  assert.match(
    pluginSource,
    /Promise\.resolve\(ctx\.storage\?\.get\?\.\('selected-roster-bot-v1'\)\)[\s\S]{0,400}?\.finally\(\(\) => \$selectedRosterHydrated\.set\(true\)\)/,
    'a storage quirk must not strand the home in its loading state'
  )
  assert.match(pluginSource, /\} catch \{\s*\n\s*\/\* no storage[^\n]*\n\s*\$selectedRosterHydrated\.set\(true\)/)
})

// ── first selection + reconciliation ────────────────────────────────────────

test('the first selection picks a reachable visible bot and creates nothing', () => {
  const t = load()
  t.setPluginCtx({ storage: { set: () => undefined } })
  t.$rosterHydrated.set(true)
  t.$selectedRosterHydrated.set(true)

  const sources = [{ connectionId: 'local', kind: 'local', label: 'This device', reachable: true }]
  const roster = [{ connectionId: 'local', name: 'writer' }, { connectionId: 'local', name: 'coder' }]

  t.reconcileRosterSelection(roster, sources, {})

  assert.equal(t.$selectedRosterKey.get(), 'local::writer')
  assert.equal(t.$openBotChat.get(), null, 'selection is not an open')
  assertNothingRouted(t, 'seating the first selection')
})

test('an unreachable bot is never auto-selected', () => {
  const t = load()
  t.setPluginCtx({ storage: { set: () => undefined } })
  t.$rosterHydrated.set(true)
  t.$selectedRosterHydrated.set(true)

  const sources = [
    { connectionId: 'work-vps', kind: 'remote', label: 'Work', reachable: false },
    { connectionId: 'local', kind: 'local', label: 'This device', reachable: true }
  ]
  const roster = [
    { connectionId: 'work-vps', name: 'researcher', remoteSource: true, sourceScoped: true },
    { connectionId: 'local', name: 'writer' }
  ]

  t.reconcileRosterSelection(roster, sources, {})

  assert.equal(t.$selectedRosterKey.get(), 'local::writer')
})

test('nothing is selected before the roster has answered', () => {
  const t = load()
  t.setPluginCtx({ storage: { set: () => undefined } })
  t.$rosterHydrated.set(false)
  t.$selectedRosterHydrated.set(true)

  t.reconcileRosterSelection([{ connectionId: 'local', name: 'writer' }], [], {})

  assert.equal(t.$selectedRosterKey.get(), '', 'a pending roster must not seat a selection it may have to revoke')
})

test('a hidden bot is not auto-selected', () => {
  const t = load()
  t.setPluginCtx({ storage: { set: () => undefined } })
  t.$rosterHydrated.set(true)
  t.$selectedRosterHydrated.set(true)

  const sources = [{ connectionId: 'local', kind: 'local', label: 'This device', reachable: true }]
  const roster = [{ connectionId: 'local', name: 'hidden-one' }, { connectionId: 'local', name: 'writer' }]

  t.reconcileRosterSelection(roster, sources, { 'hidden-one': { hidden: true } })

  assert.equal(
    t.$selectedRosterKey.get(),
    'local::writer',
    'the home must not open onto a bot the user removed from the roster'
  )
})

// ── offline owner survives; deleted owner does not ──────────────────────────

test('an offline gateway keeps the selection and its identity (relaunch case)', () => {
  const t = load()
  t.setPluginCtx({ storage: { set: () => undefined } })
  t.$rosterHydrated.set(true)
  t.$selectedRosterHydrated.set(true)
  t.$selectedRosterKey.set('work-vps::researcher')

  // The gateway is registered but down, so it contributes no roster rows.
  const sources = [
    { connectionId: 'work-vps', kind: 'remote', label: 'Work', reachable: false, error: 'ECONNREFUSED' },
    { connectionId: 'local', kind: 'local', label: 'This device', reachable: true }
  ]
  const roster = [{ connectionId: 'local', name: 'writer' }]

  t.reconcileRosterSelection(roster, sources, {})

  assert.equal(
    t.$selectedRosterKey.get(),
    'work-vps::researcher',
    'an unreachable owner is cached, not replaced — falling back would re-own the bot on another gateway'
  )

  const ghost = t.ghostRosterOwner('work-vps::researcher', sources)
  assert.equal(ghost.name, 'researcher')
  assert.equal(ghost.connectionId, 'work-vps')
  assert.equal(ghost.connectionLabel, 'Work')
  assert.equal(ghost.remoteSource, true)
  assert.equal(ghost.sourceReachable, false)
  assertNothingRouted(t, 'rendering an offline owner')
})

test('cold-start outage keeps the selected owner in the visible roster without caching every bot', () => {
  const t = load()
  const sources = [
    { connectionId: 'work-vps', kind: 'remote', label: 'Work', reachable: false, error: 'ECONNREFUSED' },
    { connectionId: 'local', kind: 'local', label: 'This device', reachable: true }
  ]
  const localOnly = [{ connectionId: 'local', name: 'writer' }]

  const restored = t.rosterWithSelectedOwner(localOnly, sources, 'work-vps::researcher')

  assert.equal(restored.length, 2, 'only the exact selected owner is restored, not an invented remote roster')
  assert.equal(t.botRosterKey(restored[1]), 'work-vps::researcher')
  assert.equal(restored[1].connectionLabel, 'Work')
  assert.equal(restored[1].sourceReachable, false)
  assert.equal(t.rosterWithSelectedOwner(restored, sources, 'work-vps::researcher').length, 2, 'never duplicated')
})

test('a reachable source never receives a ghost row for a deleted bot', () => {
  const t = load()
  const roster = [{ connectionId: 'local', name: 'writer' }]
  const sources = [
    { connectionId: 'work-vps', kind: 'remote', label: 'Work', reachable: true },
    { connectionId: 'local', kind: 'local', label: 'This device', reachable: true }
  ]

  assert.equal(t.rosterWithSelectedOwner(roster, sources, 'work-vps::deleted').length, 1)
})

test('a reachable source that no longer lists the bot clears the selection', () => {
  const t = load()
  t.setPluginCtx({ storage: { set: () => undefined } })
  t.$rosterHydrated.set(true)
  t.$selectedRosterHydrated.set(true)
  t.$selectedRosterKey.set('local::deleted-bot')

  const sources = [{ connectionId: 'local', kind: 'local', label: 'This device', reachable: true }]
  const roster = [{ connectionId: 'local', name: 'writer' }]

  assert.equal(t.ghostRosterOwner('local::deleted-bot', sources), null, 'a live source answering without it is proof')

  t.reconcileRosterSelection(roster, sources, {})

  assert.equal(t.$selectedRosterKey.get(), 'local::writer', 'the invalid selection is replaced, not kept')
})

test('an unknown source list is not proof of deletion', () => {
  const t = load()

  // Sources have not hydrated yet — the owner must keep its identity.
  assert.ok(t.ghostRosterOwner('work-vps::researcher', []))
  assert.equal(t.ghostRosterOwner('work-vps::researcher', [{ connectionId: 'local', reachable: true }]), null)
})

// ── explicit open: every reachable owner routes exactly ────────────────────

test('opening a remote bot selects and opens its exact owner chat', async () => {
  const t = load()
  t.setPluginCtx({ storage: { set: () => undefined } })
  t.$botsPaneVisible.set(true)

  const bot = {
    connectionId: 'work-vps',
    connectionLabel: 'Work',
    name: 'researcher',
    remoteSource: true,
    sourceScoped: true
  }

  const result = await t.openRosterBot(bot)

  assert.equal(result, true)
  assert.equal(t.$selectedRosterKey.get(), 'work-vps::researcher')
  const openBotChat = t.$openBotChat.get()
  assert.equal(openBotChat?.key, 'work-vps::researcher')
  assert.equal(openBotChat?.openedRegistryId, 'chat-work-vps-researcher')
  assert.equal(t.sessionOpens.length, 1)
  assert.equal(t.sessionOpens[0].options.workspaceMode, 'bots')
  assert.equal(t.sessionOpens[0].options.workspaceOwnerKey, 'bot:work-vps::researcher')
  assert.equal(t.sessionOpens[0].options.tabTitle, 'Bot Chat')
  assert.equal(t.sessionOpens[0].options.keepAllProfilesScope, true)
  assert.equal(t.sessionOpens[0].options.route.connectionId, 'work-vps')
  assert.ok(t.requests.every(request => request.route?.connectionId === 'work-vps'))
})

test('a remote owner opens its chat without closing an unrelated group tab', async () => {
  const t = load()
  t.setPluginCtx({ storage: { set: () => undefined } })
  t.$botsPaneVisible.set(true)
  t.openGroupChat('Launch room')
  const groupEntry = t.opened.find(entry => entry.id === 'hermes-bots:group:launch-room')

  const result = await t.openRosterBot({
    connectionId: 'work-vps',
    connectionLabel: 'Work',
    name: 'researcher',
    remoteSource: true
  })

  assert.equal(result, true)
  assert.equal(t.botsHomeVisible(), false)
  assert.equal(t.$groupChatWorkspace.get(), null)
  assert.equal(groupEntry.disposed, false, 'explicit selection must not close an unrelated group tab')
  assert.equal(t.sessionOpens.length, 1)
})

test('a remote owner does not depend on the informational home surface', async () => {
  const t = load()
  t.setPluginCtx({ storage: { set: () => undefined } })
  t.$botsPaneVisible.set(true)
  t.openGroupChat('Launch room')
  const groupEntry = t.opened.find(entry => entry.id === 'hermes-bots:group:launch-room')
  t.host.openWorkspace = () => {
    throw new Error('workspace unavailable')
  }

  const result = await t.openRosterBot({
    connectionId: 'work-vps',
    connectionLabel: 'Work',
    name: 'researcher',
    remoteSource: true
  })

  assert.equal(result, true)
  assert.equal(t.$groupChatWorkspace.get(), null)
  assert.equal(groupEntry.disposed, false)
  assert.equal(t.sessionOpens.length, 1)
})

test('a failed local open leaves no phantom owner in the center', async () => {
  const t = load()
  t.setPluginCtx({ storage: { set: () => undefined } })
  t.$botsPaneVisible.set(true)
  t.$openBotChat.set({ key: 'local::writer', openedRegistryId: 'previous' })

  // A source-scoped row on a desktop that cannot address it: prepareBotSource
  // refuses rather than letting the open fall through to the live gateway.
  delete t.host.requestProfile
  const bot = { connectionId: 'work-vps', name: 'writer', sourceScoped: true }

  const result = await t.openRosterBot(bot)

  assert.equal(result, false)
  assert.equal(t.$openBotChat.get(), null, 'a failed open must release the center back to the home')
  assert.equal(t.notifications.at(-1).kind, 'error', 'and the failure is surfaced, not swallowed')
  assertNothingRouted(t, 'a refused local open')
})

test('an older gateway gets an actionable Bot Mode update message', async () => {
  const t = load()
  t.setPluginCtx({ storage: { set: () => undefined } })
  t.$botsPaneVisible.set(true)
  t.host.requestProfile = async () => {
    throw new Error('Unknown method profiles.list')
  }

  const result = await t.openRosterBot({
    connectionId: 'work-vps',
    connectionLabel: 'Work',
    name: 'writer',
    sourceScoped: true
  })

  assert.equal(result, false)
  assert.equal(t.notifications.at(-1).title, 'Update this gateway to use Bot Mode')
  assert.equal(t.notifications.at(-1).message, 'Update Work, then try again.')
})

test('a missing profile-scoped draft API returns to the home without navigating', async () => {
  const t = load()
  t.setPluginCtx({ storage: { set: () => undefined } })
  t.$botsPaneVisible.set(true)
  t.host.request = async method => {
    if (method === 'session.list') return { sessions: [] }
    if (method === 'session.create') return {}
    return {}
  }

  const result = await t.openRosterBot({ connectionId: 'local', name: 'writer' })

  assert.equal(result, false)
  assert.equal(t.$openBotChat.get(), null, 'no draft was opened without the owner-scoped API')
  assert.ok(t.botsHomeVisible(), 'the owner home remains the visible recovery surface')
})

test('a local bot chat retires the prior group tab before taking the center', async () => {
  const t = load()
  t.setPluginCtx({ storage: { set: () => undefined } })
  t.$botsPaneVisible.set(true)
  t.$groupChats.set({ 'Launch room': { roomId: 'r-launch', log: [], watermarks: {}, sessions: {} } })
  t.host.openSession = async () => undefined
  t.host.request = async method => {
    if (method === 'session.list') {
      return { sessions: [{ id: 'bot-chat', title: 'Bot Chat', message_count: 4 }] }
    }

    return {}
  }

  t.openGroupChat('Launch room')
  const groupEntry = t.opened.find(entry => entry.id === 'hermes-bots:group:launch-room')
  assert.ok(groupEntry)
  assert.equal(t.$groupChatWorkspace.get(), 'Launch room')

  const result = await t.openRosterBot({ connectionId: 'local', name: 'writer' })

  assert.equal(result, true)
  assert.equal(t.$groupChatWorkspace.get(), null)
  assert.equal(groupEntry.disposed, true, 'the local bot chat must release the group workspace owner')
  assert.equal(t.$openBotChat.get()?.key, 'local::writer')
  assert.equal(t.$openBotChat.get()?.openedRegistryId, 'bot-chat')
})

test('a failed canonical-chat open preserves the visible group owner', async () => {
  const t = load()
  t.setPluginCtx({ storage: { set: () => undefined } })
  t.$botsPaneVisible.set(true)
  t.$groupChats.set({ 'Launch room': { roomId: 'r-launch', log: [], watermarks: {}, sessions: {} } })
  t.host.openSession = async () => {
    throw new Error('gateway unavailable')
  }
  t.host.request = async method => {
    if (method === 'session.list') {
      return { sessions: [{ id: 'bot-chat', title: 'Bot Chat', message_count: 4 }] }
    }

    return {}
  }

  t.openGroupChat('Launch room')
  const groupEntry = t.opened.find(entry => entry.id === 'hermes-bots:group:launch-room')
  const result = await t.openRosterBot({ connectionId: 'local', name: 'writer' })

  assert.equal(result, false)
  assert.equal(t.$groupChatWorkspace.get(), 'Launch room')
  assert.equal(groupEntry.disposed, true, 'the stale group tab was retired before the attempted open')
  const restoredGroup = t.opened.filter(entry => entry.id === 'hermes-bots:group:launch-room').at(-1)
  assert.notEqual(restoredGroup, groupEntry)
  assert.equal(restoredGroup.disposed, false, 'a failed transition re-registers the group tab')
  assert.equal(t.$openBotChat.get(), null)
  assert.equal(t.notifications.at(-1).kind, 'error')
})

test('a failed remote canonical open re-fronts the preserved group tab', async () => {
  const t = load()
  t.setPluginCtx({ storage: { set: () => undefined } })
  t.$botsPaneVisible.set(true)
  t.$groupChats.set({ 'Launch room': { roomId: 'r-launch', log: [], watermarks: {}, sessions: {} } })
  t.openGroupChat('Launch room')
  const groupEntry = t.opened.find(entry => entry.id === 'hermes-bots:group:launch-room')
  t.host.openSession = async () => {
    throw new Error('hydration failed after fronting Bot Chat')
  }

  const result = await t.openRosterBot({
    connectionId: 'work-vps',
    connectionLabel: 'Work',
    name: 'researcher',
    remoteSource: true
  })

  assert.equal(result, false)
  assert.equal(groupEntry.disposed, false, 'remote open preserves the original tab')
  assert.equal(t.$groupChatWorkspace.get(), 'Launch room')
  assert.equal(
    t.opened.filter(entry => entry.id === 'hermes-bots:group:launch-room').length,
    2,
    're-opening the same workspace id re-fronts the preserved group tab'
  )
})

test('a failed remote open cannot restore a group disbanded while the tab was preserved', async () => {
  const t = load()
  t.setPluginCtx({ storage: { set: () => undefined } })
  t.$botsPaneVisible.set(true)
  t.$groupChats.set({ 'Launch room': { roomId: 'r-launch', log: [], watermarks: {}, sessions: {} } })
  t.openGroupChat('Launch room')

  let rejectOpen
  let markOpenStarted
  const openStarted = new Promise(resolve => {
    markOpenStarted = resolve
  })
  t.host.openSession = () =>
    new Promise((resolve, reject) => {
      rejectOpen = reject
      markOpenStarted()
    })

  const opening = t.openRosterBot({
    connectionId: 'work-vps',
    connectionLabel: 'Work',
    name: 'researcher',
    remoteSource: true
  })
  await openStarted

  t.$groupChats.set({ 'Launch room': { roomId: 'r-launch', tombstone: true } })
  t.closeGroupChatMainTab('Launch room')
  rejectOpen(new Error('gateway unavailable'))

  assert.equal(await opening, false)
  assert.equal(t.$groupChatWorkspace.get(), null)
  assert.equal(t.$openBotChat.get(), null)
})

test('a failed local open cannot re-register a group tombstoned after retirement', async () => {
  const t = load()
  t.setPluginCtx({ storage: { set: () => undefined } })
  t.$botsPaneVisible.set(true)
  t.$groupChats.set({ 'Launch room': { roomId: 'r-launch', log: [], watermarks: {}, sessions: {} } })
  t.host.request = async method => {
    if (method === 'session.list') {
      return { sessions: [{ id: 'bot-chat', title: 'Bot Chat', message_count: 4 }] }
    }

    return {}
  }

  let rejectOpen
  let markOpenStarted
  const openStarted = new Promise(resolve => {
    markOpenStarted = resolve
  })
  t.host.openSession = () =>
    new Promise((resolve, reject) => {
      rejectOpen = reject
      markOpenStarted()
    })

  t.openGroupChat('Launch room')
  const opening = t.openRosterBot({ connectionId: 'local', name: 'writer' })
  await openStarted

  t.$groupChats.set({ 'Launch room': { roomId: 'r-launch', tombstone: true } })
  rejectOpen(new Error('gateway unavailable'))

  assert.equal(await opening, false)
  assert.equal(t.$groupChatWorkspace.get(), null)
  assert.equal(t.opened.filter(entry => entry.id === 'hermes-bots:group:launch-room').length, 1)
  assert.equal(t.$openBotChat.get(), null)
})

test('switching bots does not let the home cover the current chat while the next chat opens', async () => {
  const t = load({ focusedStoredSessionId: 'chat-alpha' })
  t.setPluginCtx({ storage: { set: () => undefined } })
  t.$botsPaneVisible.set(true)
  t.$openBotChat.set({
    key: 'local::alpha',
    openedRegistryId: 'chat-alpha',
    openedSessionId: 'chat-alpha'
  })
  t.host.request = async method => {
    if (method === 'session.list') {
      return { sessions: [{ id: 'chat-beta', title: 'Bot Chat', message_count: 4 }] }
    }

    return {}
  }

  let finishOpen
  let markOpenStarted
  const openStarted = new Promise(resolve => {
    markOpenStarted = resolve
  })
  t.host.openSession = () =>
    new Promise(resolve => {
      finishOpen = resolve
      markOpenStarted()
    })

  const opening = t.openRosterBot({ connectionId: 'local', name: 'beta' })
  await openStarted

  // The destination focus edge releases the old chat claim before the async
  // open has returned its replacement claim. Passive workspace reconciliation
  // must not use that handoff gap to put Bots home in front.
  t.focused.set(null)
  t.releaseStaleOpenBotChat(null)
  t.syncBotsHomeWorkspace()

  assert.equal(t.opened.some(entry => entry.id === 'hermes-bots:home'), false)

  finishOpen()
  assert.equal(await opening, true)
  assert.equal(t.$openBotChat.get()?.openedRegistryId, 'chat-beta')
})

test('choosing a group prevents a stale canonical-chat open from closing it later', async () => {
  const t = load()
  t.setPluginCtx({ storage: { set: () => undefined } })
  t.$botsPaneVisible.set(true)

  let finishOpen
  let markOpenStarted
  const openStarted = new Promise(resolve => {
    markOpenStarted = resolve
  })
  t.host.openSession = () =>
    new Promise(resolve => {
      finishOpen = resolve
      markOpenStarted()
    })
  t.host.request = async method => {
    if (method === 'session.list') {
      return { sessions: [{ id: 'bot-chat', title: 'Bot Chat', message_count: 4 }] }
    }

    return {}
  }

  const opening = t.openRosterBot({ connectionId: 'local', name: 'writer' })
  await openStarted
  t.openGroupChat('Launch room')
  const groupEntry = t.opened.find(entry => entry.id === 'hermes-bots:group:launch-room')

  finishOpen()
  assert.equal(await opening, false)
  assert.equal(t.$groupChatWorkspace.get(), 'Launch room')
  assert.equal(groupEntry.disposed, false)
  assert.equal(t.$openBotChat.get(), null)
})

// ── who owns the main workspace ─────────────────────────────────────────────

test('ownership table: exactly one surface owns the center', () => {
  const t = load()

  // Bot Mode not on screen: neither surface exists.
  t.$botsPaneVisible.set(false)
  assert.equal(t.botsHomeMayOpen(false), false)
  assert.equal(t.botChatOwnsWorkspace(), false)

  // Bots visible, nothing else: the home owns it, Cronjobs stay away.
  t.$botsPaneVisible.set(true)
  assert.equal(t.botsHomeMayOpen(false), true)
  assert.equal(t.botChatOwnsWorkspace(), false, 'no bot chat owns the center, so bot-scoped Cronjobs must not seat')

  // A group chat owns it: neither the home nor Cronjobs.
  t.$groupChatWorkspace.set('Core')
  assert.equal(t.botsHomeMayOpen(false), false)
  assert.equal(t.botsHomeMayOpen(true), false, 'even an explicit gesture cannot cover a group chat')
  assert.equal(t.botChatOwnsWorkspace(), false)
  t.$groupChatWorkspace.set(null)

  // A bot chat owns it: the home yields and Cronjobs seat.
  t.$openBotChat.set({ key: 'local::writer', openedRegistryId: 'chat-1' })
  assert.equal(t.botsHomeMayOpen(false), false)
  assert.equal(t.botChatOwnsWorkspace(), true)
  t.$openBotChat.set(null)

  // A focused session (restored at boot, or Sessions mode) vetoes PASSIVE
  // opens but not an explicit gesture at the home.
  t.focused.set('chat-9')
  assert.equal(t.sessionOwnsWorkspace(), true)
  assert.equal(t.botsHomeMayOpen(false), false)
  assert.equal(t.botsHomeMayOpen(true), true)
  assert.equal(t.botChatOwnsWorkspace(), true)
})

test('with the home tab fronted, the hidden chat does not seat Cronjobs', () => {
  const t = load()
  t.$botsPaneVisible.set(true)

  // Explicitly front the home over a focused chat.
  t.focused.set('chat-1')
  t.openBotsHomeWorkspace(true)
  assert.equal(t.botsHomeVisible(), true)
  assert.equal(
    t.botChatOwnsWorkspace(),
    false,
    'the chat is a hidden sibling layer while the home holds the tab slot'
  )

  // The user fronts the chat tab again (no focus change fires): the pane
  // visibility flip alone must reseat Cronjobs.
  t.closeBotsHomeWorkspace()
  assert.equal(t.botsHomeVisible(), false)
  assert.equal(t.botChatOwnsWorkspace(), true)
})

test('a persisted layout that restored the home behind the draft gets re-fronted', () => {
  const t = load()
  t.$botsPaneVisible.set(true)
  t.syncBotsHomeWorkspace()
  assert.equal(t.opened.length, 1)

  // The tree restored the tab BEHIND the core draft pane (adoption kept the
  // persisted active slot): the ownerless composer would sit on top.
  t.paneVisible.set('plugin-workspace:hermes-bots:home', false)

  t.syncBotsHomeWorkspace()

  assert.equal(t.opened.length, 2, 're-opened to reclaim the active slot')
  assert.equal(t.closed.length, 1, 'the stale registration was closed first — never two live disposers')
  assert.equal(t.botsHomeVisible(), true)
})

test('an explicit remote selection opens its owner tab without moving the focused Sessions chat', async () => {
  const t = load({ focusedStoredSessionId: 'local-scout-chat' })
  t.setPluginCtx({ storage: { set: () => undefined } })
  t.$botsPaneVisible.set(true)

  // Passive sync must not cover the chat…
  t.syncBotsHomeWorkspace()
  assert.deepEqual(t.opened, [])

  // …but clicking the same-named twin on another gateway opens that owner in
  // Bot Mode while the Sessions chat stays alive underneath.
  await t.openRosterBot({ connectionId: 'work-vps', connectionLabel: 'Work', name: 'scout', remoteSource: true })

  assert.equal(t.opened.length, 0)
  assert.equal(t.sessionOpens.length, 1)
  assert.equal(t.$selectedRosterKey.get(), 'work-vps::scout')
  assert.equal(t.$openBotChat.get().key, 'work-vps::scout')
  assert.equal(t.focused.get(), 'local-scout-chat')

  // Browsing more remote owners opens that owner without reusing the first.
  await t.openRosterBot({ connectionId: 'work-vps', connectionLabel: 'Work', name: 'relay', remoteSource: true })
  assert.equal(t.sessionOpens.length, 2)
  assert.equal(t.$selectedRosterKey.get(), 'work-vps::relay')
})

test('an explicit Bots-home gesture fronts the selected owner over a Sessions composer', () => {
  const t = load({ focusedStoredSessionId: 'sessions-chat' })
  t.$botsPaneVisible.set(true)

  t.syncBotsHomeWorkspace()
  assert.deepEqual(t.opened, [], 'passive polling still leaves a focused session alone')

  t.openBotsHomeWorkspace(true)
  assert.equal(t.opened.length, 1, 'the explicit gesture has an exact owner instead of the Sessions composer')
  assert.equal(t.opened[0].id, 'hermes-bots:home')
})

test('source contract: sidebar entry and boot restore reconcile passively after layout hydration', () => {
  assert.match(pluginSource, /const syncWorkspaceSurfaces = \(\) =>/)
  assert.match(pluginSource, /stopSidebarSync = \$sidebarVisible\.listen\(visible => \{[\s\S]{0,1500}?syncWorkspaceSurfaces\(\)/)
  assert.doesNotMatch(pluginSource, /stopSidebarSync = \$sidebarVisible\.listen\(visible => \{[\s\S]{0,1500}?syncWorkspaceSurfaces\(Boolean\(visible\)\)/)
  assert.match(
    pluginSource,
    /\$botChatFocused\.set\(sessionOwnsWorkspace\(\)\)[\s\S]{0,500}?syncWorkspaceSurfaces\(\)[\s\S]{0,120}?scheduleSurfaceSync\(\)/
  )
  assert.match(
    pluginSource,
    /homeVisibleStore\.listen\(visible => \{[\s\S]{0,260}?\$botsHomeFronted\.set\(Boolean\(visible\)\)[\s\S]{0,120}?scheduleSurfaceSync\(\)/
  )
})

test('an opened chat releases the center once focus leaves it', () => {
  const t = load()
  t.$openBotChat.set({ key: 'local::writer', openedRegistryId: 'chat-1' })

  t.releaseStaleOpenBotChat('chat-1')
  assert.deepEqual(t.$openBotChat.get(), { key: 'local::writer', openedRegistryId: 'chat-1' }, 'still the focused chat')

  t.releaseStaleOpenBotChat('chat-2')
  assert.equal(t.$openBotChat.get(), null, 'another session took the center')

  t.$openBotChat.set({ key: 'local::writer', openedRegistryId: 'chat-1' })
  t.releaseStaleOpenBotChat(null)
  assert.equal(t.$openBotChat.get(), null, 'the chat was closed — the home may come back')
})

test('a legacy draft keeps the center until a real session takes focus', () => {
  const t = load()

  // The newChat fallback has no stored id to compare against.
  t.$openBotChat.set({ key: 'local::writer', openedRegistryId: '' })

  t.releaseStaleOpenBotChat(null)
  assert.deepEqual(t.$openBotChat.get(), { key: 'local::writer', openedRegistryId: '' }, 'an unsent draft is still that bot’s')

  t.releaseStaleOpenBotChat('chat-7')
  assert.equal(t.$openBotChat.get(), null)
})

// ── the home tab itself ─────────────────────────────────────────────────────

test('the home opens once and is not re-fronted while it already owns the center', () => {
  const t = load()
  t.$botsPaneVisible.set(true)

  t.syncBotsHomeWorkspace()
  assert.equal(t.opened.length, 1)
  assert.equal(t.opened[0].id, 'hermes-bots:home')
  assert.equal(t.opened[0].options.title, 'Bots')

  // Repeated signals (focus churn, roster polls) must not steal focus back
  // or mint a second disposer whose stale predecessor could tear down the
  // newer registration.
  t.syncBotsHomeWorkspace()
  t.syncBotsHomeWorkspace()
  assert.equal(t.opened.length, 1)
  assert.deepEqual(t.closed, [])
})

test('the home yields the center to a chat and returns when the chat closes', () => {
  const t = load()
  t.$botsPaneVisible.set(true)

  t.syncBotsHomeWorkspace()
  assert.equal(t.opened.length, 1)

  t.$openBotChat.set({ key: 'local::writer', openedRegistryId: 'chat-1' })
  t.syncBotsHomeWorkspace()
  assert.equal(t.closed.length, 1, 'the home closed for the chat')

  t.$openBotChat.set(null)
  t.syncBotsHomeWorkspace()
  assert.equal(t.opened.length, 2, 'and comes back when nothing else owns the center')
})

test('a restored session at boot keeps the home from covering it', () => {
  const t = load({ focusedStoredSessionId: 'restored-chat' })
  t.$botsPaneVisible.set(true)

  t.syncBotsHomeWorkspace()

  assert.deepEqual(t.opened, [], 'the home must not steal the tab from a session the user left open')
})

test('closing the home tab does not resurrect it mid-close', () => {
  const t = load()
  t.$botsPaneVisible.set(true)
  t.syncBotsHomeWorkspace()

  // The tab's own ✕ routes through the same disposer.
  t.opened[0].options.onClose()

  assert.equal(t.opened.length, 1, 'onClose must not re-open the tab the user just closed')

  // It comes back only when something else actually happens.
  t.syncBotsHomeWorkspace()
  assert.equal(t.opened.length, 2)
})

test('the home never yanks the center back from a sibling tab the user chose', () => {
  const t = load()
  t.$botsPaneVisible.set(true)
  t.syncBotsHomeWorkspace()
  assert.equal(t.opened.length, 1)

  // The user tabs to the ordinary draft workspace: no session owns the
  // center, so the home still "should" own it — but it is already open, and
  // re-opening would front it over the tab the user just picked.
  t.focused.set(null)
  t.syncBotsHomeWorkspace()
  t.syncBotsHomeWorkspace()

  assert.equal(t.opened.length, 1, 'focus churn must not re-front the home')
  assert.deepEqual(t.closed, [])
})

test('older shells without owner routing fail with an update path', async () => {
  const t = load({ openWorkspace: false, paneVisibility: false })
  t.setPluginCtx({ storage: { set: () => undefined } })
  t.$botsPaneVisible.set(true)
  delete t.host.requestProfile
  delete t.host.openSession

  t.syncBotsHomeWorkspace()
  assert.deepEqual(t.opened, [])

  // A remote row cannot guess through the active gateway on an old shell.
  await t.openRosterBot({
    connectionId: 'work-vps',
    connectionLabel: 'Work',
    name: 'researcher',
    remoteSource: true,
    sourceScoped: true
  })

  assert.equal(t.notifications.length, 1)
  assert.match(t.notifications[0].message, /Could not reach Work/)
  assert.match(String(t.notifications[0].error), /Update Hermes Desktop/)
  assertNothingRouted(t, 'remote row on an older shell')
})

// ── view contracts ──────────────────────────────────────────────────────────

test('the home Tip wraps exactly one element (Radix asChild)', () => {
  const start = pluginSource.indexOf('function BotsHomeView(')
  assert.ok(start >= 0)
  const view = pluginSource.slice(start, pluginSource.indexOf('function closeBotsHomeWorkspace('))

  assert.match(view, /jsx\(Tip, \{\s*\n\s*label:[^\n]*\n\s*children: jsxs\('div'/, 'one child element, not an array')
  assert.doesNotMatch(view, /jsxs\(Tip, \{/, 'jsxs would pass multiple children and break the trigger')
  // The screen-reader text rides INSIDE the trigger element.
  assert.match(view, /className: 'sr-only'/)
})

test('the home shows a loading state rather than flashing “No bots”', () => {
  const start = pluginSource.indexOf('function BotsHomeView(')
  const view = pluginSource.slice(start, pluginSource.indexOf('function closeBotsHomeWorkspace('))

  assert.match(view, /if \(!rosterHydrated \|\| !selectionHydrated\) \{[\s\S]{0,220}?GlyphSpinner/)
  const spinnerAt = view.indexOf('GlyphSpinner')
  const emptyAt = view.indexOf("title: roster.length ? 'Choose a bot or group chat' : 'No bots yet'")
  assert.ok(spinnerAt >= 0 && emptyAt > spinnerAt, 'the empty state is only reachable after both hydrations')
})

test('the home uses the neutral workspace surface instead of a transient blue tint', () => {
  const start = pluginSource.indexOf('function BotsHomeView(')
  const view = pluginSource.slice(start, pluginSource.indexOf('function closeBotsHomeWorkspace('))

  assert.match(view, /className: 'flex h-full min-h-0 flex-col bg-background'/)
  assert.doesNotMatch(view, /bg-\(--ui-bg-primary\)/)
})

test('roster hydration and selection reconciliation run after render', () => {
  const start = pluginSource.indexOf('function BotsPane(')
  const pane = pluginSource.slice(start, pluginSource.indexOf('// ── registration'))

  assert.match(
    pane,
    /useEffect\(\(\) => \{[\s\S]{0,500}?\$rosterHydrated\.set\(true\)[\s\S]{0,300}?reconcileRosterSelection\(roster, sourceSnapshot, allMeta\)[\s\S]{0,700}?\}, \[data, error, selectionHydrated, roster, sourceSnapshot, allMeta\]\)/,
    'persisted roster ownership must reconcile from an effect, never from a replayable render'
  )
  assert.equal(
    (pane.match(/reconcileRosterSelection\(roster, sourceSnapshot, allMeta\)/g) || []).length,
    1,
    'BotsPane has one effect-bound reconciliation path'
  )
  assert.match(
    pane,
    /sourceWithSelectedOwner = selectionHydrated && rosterHydrated[\s\S]{0,180}?rosterWithSelectedOwner/,
    'an unavailable-owner placeholder cannot bypass initial roster hydration'
  )
  assert.match(
    pane,
    /\$lastRoster\.set\(roster\.filter\(row => !row\?\.ghost\)\)/,
    'presentation-only owner placeholders never enter shared roster state'
  )
})

test('an unavailable owner offers retry instead of a dead Open chat button', () => {
  const start = pluginSource.indexOf('function BotsHomeView(')
  const view = pluginSource.slice(start, pluginSource.indexOf('function closeBotsHomeWorkspace('))

  assert.match(view, /unavailable && !sourceRemoved\s*\n\s*\? jsx\(Button, \{[\s\S]{0,200}?children: 'Retry'/)
  // Retry re-polls the roster; it must not activate or route anything.
  assert.match(view, /queryClient\.invalidateQueries\(\{ queryKey: ROSTER_KEY \}\)/)
  assert.doesNotMatch(view, /ensureAgent|requestProfile|newChat/)
  assert.match(view, /\$\{gateway\} is unavailable\. Retry when it is back online\./)
  assert.match(view, /\$\{gateway\} was removed\. Choose another bot from the sidebar\./)
  assert.doesNotMatch(view, /This bot remains selected/)
  assert.doesNotMatch(view, /its work keeps running on that gateway/)
})

test('an available remote owner offers the same direct chat action', () => {
  const start = pluginSource.indexOf('function BotsHomeView(')
  const view = pluginSource.slice(start, pluginSource.indexOf('function closeBotsHomeWorkspace('))

  assert.match(view, /children: 'Open chat'/)
  assert.match(view, /Open this bot’s continuous chat/)
  assert.doesNotMatch(view, /Copy @/)
  assert.doesNotMatch(view, /remoteCopy/)
  assert.doesNotMatch(view, /Mention it from any Bot Chat/)
})

test('an unavailable owner never presents a guessed mention handle', () => {
  const start = pluginSource.indexOf('function BotsHomeView(')
  const view = pluginSource.slice(start, pluginSource.indexOf('function closeBotsHomeWorkspace('))

  assert.match(view, /const handle = bot\.ghost \? '' : botHandle\(bot\.name, bot\)/)
  assert.match(view, /handle\s*\n\s*\? jsx\('span'/)
})

test('Bot Mode copy says bot, not agent', () => {
  assert.doesNotMatch(pluginSource, /Name the agent first/)
  assert.doesNotMatch(pluginSource, /create agents first/)
  assert.doesNotMatch(pluginSource, /children: busy \? 'Creating…' : 'Create Agent'/)
  assert.doesNotMatch(pluginSource, /`Agent "\$\{displayName\(\{ name: slug, title \}\)\}" created/)
  assert.match(pluginSource, /Name the bot first/)
  assert.match(pluginSource, /children: busy \? 'Creating…' : 'Create Bot'/)
  assert.match(pluginSource, /`Bot "\$\{displayName\(\{ name: slug, title \}\)\}" created/)
})
