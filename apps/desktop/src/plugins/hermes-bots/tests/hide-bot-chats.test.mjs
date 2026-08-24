import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

// Bot Mode sessions are ALWAYS hidden from the global Sessions sidebar
// (canonical Bot Chats and group-chat member sessions alike) via the core
// generic `hidden` session flag. There is no user pref: session.create
// passes hidden:true unconditionally, and hideOwnedBotSessions() sweeps
// every known plugin-owned session id through session.set_hidden so rows
// born visible under the old pref get reconciled.

const source = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

function loadCreate() {
  const start = source.indexOf('const canonicalCreations = new Map()')
  const end = source.indexOf('function displayName(', start)
  const created = []
  const context = {
    host: {
      openSession: async () => {},
      request: async (method, params) => {
        if (method === 'session.create') {
          created.push(params)
          return { stored_session_id: 'sid-1', session_id: 'rt-1' }
        }
        return {}
      }
    },
    saveBotMeta: () => {},
    backendTargetProfile: (route, name) => route?.targetProfile || name,
    botOwner: name => ({ bot: { name }, key: name, name, route: null }),
    requestForBot: (_bot, method, params) => context.host.request(method, params),
    window: { setTimeout: cb => cb() }
  }
  const section = source.slice(start, end).concat('\nglobalThis.__c = { createCanonicalChat };\n')
  vm.runInNewContext(section, context, { filename: 'c.js' })
  return { create: context.__c.createCanonicalChat, created }
}

test('createCanonicalChat always passes hidden:true — no pref gate', async () => {
  const { create, created } = loadCreate()
  await create('alpha')
  assert.equal(created.length, 1)
  assert.equal(created[0].hidden, true)
  assert.equal(created[0].title, 'Bot Chat')
})

test('group member session.create is unconditionally hidden too', () => {
  // Source contract on ensureGroupChatSession: the create carries a literal
  // hidden:true, with no $hideBotChats conditional anywhere in the plugin.
  const fn = source.slice(source.indexOf('async function ensureGroupChatSession('), source.indexOf('const GROUP_TURN_TIMEOUT_MS'))
  assert.match(fn, /hidden: true/)
  assert.equal(source.includes('$hideBotChats'), false, 'the old pref atom must be gone')
})

test('hideOwnedBotSessions sweeps room member sessions by id', async () => {
  const start = source.indexOf('function hideOwnedBotSessions()')
  const end = source.indexOf('/** Fetch server-side avatars', start)
  const calls = []
  const context = {
    host: {
      request: async (method, params) => {
        calls.push({ method, params })
        return {}
      }
    },
    $botMeta: { get: () => ({}) },
    $lastRoster: { get: () => [] },
    backendTargetProfile: (route, name) => route?.targetProfile || name,
    botConnectionRoute: () => null,
    botMetaKey: bot => bot.name,
    requestForBot: (_bot, method, params) => context.host.request(method, params),
    $groupChats: {
      get: () => ({
        Core: { sessions: { alpha: 'room-core-a', beta: 'room-core-b' } },
        Quiet: { sessions: { alpha: 'room-core-a' } }, // duplicate id — must dedupe
        Legacy: {} // pre-sessions room shape
      })
    },
    sweepBotProfileSessions: async () => undefined
  }
  const section = source.slice(start, end).concat('\nglobalThis.__h = { hideOwnedBotSessions };\n')
  vm.runInNewContext(section, context, { filename: 'h.js' })
  await context.__h.hideOwnedBotSessions()

  const ids = calls.filter(c => c.method === 'session.set_hidden').map(c => c.params.session_id).sort()
  assert.deepEqual(ids, ['room-core-a', 'room-core-b'])
  const hiddenCalls = calls.filter(c => c.method === 'session.set_hidden')
  assert.ok(hiddenCalls.every(c => c.params.hidden === true))
})

test('hideOwnedBotSessions never consults stored canonical pointers', () => {
  // Canonical Bot Chats are hidden by the TITLE sweep (they are identified by
  // name, not by pointer) — the load-time reconciliation must not read
  // $botMeta chat ids or verify them via profiles.list.
  const start = source.indexOf('function hideOwnedBotSessions()')
  const end = source.indexOf('// Titles Bot Mode itself mints', start)
  const section = source.slice(start, end)
  assert.doesNotMatch(section, /botMeta/)
  assert.doesNotMatch(section, /profiles\.list/)
})

test('remote group member sessions derive their immutable owner from persisted room members', async () => {
  const start = source.indexOf('function hideOwnedBotSessions()')
  const end = source.indexOf('/** Fetch server-side avatars', start)
  const ambient = []
  const routed = []
  const owner = {
    name: 'worker',
    sourceScoped: true,
    route: {
      connectionId: 'source-a',
      mode: 'remote',
      profile: 'worker',
      targetProfile: 'backend-worker'
    }
  }
  const context = {
    host: { request: async (method, params) => ambient.push({ method, params }) },
    $botMeta: { get: () => ({}) },
    $lastRoster: { get: () => [] },
    $groupChats: {
      get: () => ({
        Core: {
          sessions: { 'source-a::worker': 'remote-room-1' },
          members: [owner]
        }
      })
    },
    groupMemberKey: member => `${member.route.connectionId}::${member.name}`,
    requestForBot: async (bot, method, params) => routed.push({ bot, method, params }),
    sweepBotProfileSessions: async () => undefined
  }
  const section = source.slice(start, end).concat('\nglobalThis.__h = { hideOwnedBotSessions };\n')
  vm.runInNewContext(section, context, { filename: 'h-remote.js' })

  await context.__h.hideOwnedBotSessions()

  assert.equal(ambient.some(call => call.method === 'session.set_hidden'), false)
  assert.equal(routed.length, 1)
  assert.equal(routed[0].bot.route.connectionId, 'source-a')
  assert.equal(routed[0].bot.route.targetProfile, 'backend-worker')
  assert.equal(routed[0].params.session_id, 'remote-room-1')
})

test('same session id on two remote group owners never hides an ambient collision', async () => {
  const start = source.indexOf('function hideOwnedBotSessions()')
  const end = source.indexOf('/** Fetch server-side avatars', start)
  const ambient = []
  const routed = []
  const owner = connectionId => ({
    name: 'worker',
    sourceScoped: true,
    route: { connectionId, mode: 'remote', profile: 'worker', targetProfile: 'backend-worker' }
  })
  const ownerA = owner('source-a')
  const ownerB = owner('source-b')
  const context = {
    host: { request: async (method, params) => ambient.push({ method, params }) },
    $botMeta: { get: () => ({}) },
    $lastRoster: { get: () => [] },
    $groupChats: {
      get: () => ({
        A: { sessions: { 'source-a::worker': 'same-id' }, sessionOwners: { 'source-a::worker': ownerA } },
        B: { sessions: { 'source-b::worker': 'same-id' }, sessionOwners: { 'source-b::worker': ownerB } }
      })
    },
    groupMemberKey: member => `${member.route.connectionId}::${member.name}`,
    requestForBot: async (bot, method, params) => routed.push({ bot, method, params }),
    sweepBotProfileSessions: async () => undefined
  }
  const section = source.slice(start, end).concat('\nglobalThis.__h = { hideOwnedBotSessions };\n')
  vm.runInNewContext(section, context, { filename: 'h-collision.js' })

  await context.__h.hideOwnedBotSessions()

  assert.equal(ambient.some(call => call.method === 'session.set_hidden'), false)
  assert.deepEqual(routed.map(call => call.bot.route.connectionId).sort(), ['source-a', 'source-b'])
  assert.ok(routed.every(call => call.params.session_id === 'same-id'))
})

test('malformed persisted owner for a source-qualified group session fails closed', async () => {
  const start = source.indexOf('function hideOwnedBotSessions()')
  const end = source.indexOf('/** Fetch server-side avatars', start)
  const ambient = []
  const routed = []
  const context = {
    host: { request: async (method, params) => ambient.push({ method, params }) },
    $botMeta: { get: () => ({}) },
    $lastRoster: { get: () => [] },
    $groupChats: {
      get: () => ({
        LegacyRemote: {
          sessions: { 'source-a::worker': 'same-id' },
          sessionOwners: { 'source-a::worker': { name: 'worker' } }
        }
      })
    },
    groupMemberKey: member => member?.name,
    requestForBot: async (bot, method, params) => routed.push({ bot, method, params }),
    sweepBotProfileSessions: async () => undefined
  }
  const section = source.slice(start, end).concat('\nglobalThis.__h = { hideOwnedBotSessions };\n')
  vm.runInNewContext(section, context, { filename: 'h-malformed-owner.js' })

  await context.__h.hideOwnedBotSessions()

  assert.equal(ambient.some(call => call.method === 'session.set_hidden'), false)
  assert.equal(routed.length, 0)
})

test('sweepBotProfileSessions hides Bot-Mode-titled rows per roster bot, and only those', async () => {
  // Ownership-based half of the sweep: CLI-born "Agent Inbox" / extra
  // "Bot Chat" rows live in bot profiles but are unknown to $botMeta /
  // $groupChats, so the id-based sweep never reaches them. This sweep lists
  // each roster bot's own sessions and hides rows by exact plumbing title
  // ('Bot Chat', 'Agent Inbox', 'Group: …' prefix) — never a user-titled row.
  const start = source.indexOf('function hideOwnedBotSessions()')
  const end = source.indexOf('/** Fetch server-side avatars', start)
  const calls = []
  const nowSeconds = 1_000
  const rowsByProfile = {
    alpha: [
      { id: 'a-1', title: 'Bot Chat', started_at: 1 },
      { id: 'a-2', title: 'Agent Inbox', started_at: 1 },
      { id: 'a-3', title: 'Group: Core', started_at: 1 },
      { id: 'a-4', title: 'My real conversation', started_at: 1 },
      { id: 'a-5', title: 'Bot Chat notes', started_at: 1 }, // not an exact title — kept
      { id: 'a-6', title: 'Bot Chat', started_at: nowSeconds - 299 }, // live draft — kept
      { id: 'a-7', title: 'Agent Inbox' }, // missing age metadata — kept fail-closed
      { id: 'a-8', title: 'Bot Chat', started_at: nowSeconds - 300 } // boundary reached — hidden
    ],
    remy: [{ id: 'r-1', title: 'Agent Inbox', started_at: 1 }]
  }
  const context = {
    host: { request: async () => ({}) },
    $botMeta: { get: () => ({}) },
    $groupChats: { get: () => ({}) },
    $lastRoster: { get: () => [{ name: 'alpha' }, { name: 'remy', remoteSource: true, connectionId: 'mini' }] },
    PROFILE_SESSION_LIST_LIMIT: 200,
    requestForBot: async (bot, method, params) => {
      calls.push({ bot: bot.name, method, params })
      if (method === 'session.list') {
        return { sessions: rowsByProfile[params.profile] || [] }
      }
      return {}
    }
  }
  const section = source.slice(start, end).concat('\nglobalThis.__h = { hideOwnedBotSessions, sweepBotProfileSessions };\n')
  vm.runInNewContext(section, context, { filename: 's.js' })
  await context.__h.sweepBotProfileSessions(nowSeconds)

  const lists = calls.filter(c => c.method === 'session.list')
  assert.deepEqual(lists.map(c => c.params.profile).sort(), ['alpha', 'remy'])
  // Visible-rows-only listing keeps the sweep idempotent.
  assert.ok(lists.every(c => !c.params.include_hidden))

  const hidden = calls.filter(c => c.method === 'session.set_hidden')
  assert.deepEqual(
    hidden.map(c => c.params.session_id).sort(),
    ['a-1', 'a-2', 'a-3', 'a-8', 'r-1'],
    'exact plumbing titles only — user-titled and brand-new rows stay visible'
  )
  assert.ok(hidden.every(c => c.params.hidden === true))
  // Remote-source bots route through requestForBot with their own bot row.
  assert.equal(hidden.find(c => c.params.session_id === 'r-1').bot, 'remy')
})

test('hideOwnedBotSessions chains the ownership sweep and survives its absence of context', async () => {
  // The load/reconnect entrypoint runs BOTH halves: known ids first, then
  // the roster-wide title sweep (best-effort — a throwing sweep never
  // breaks the id half).
  assert.match(source, /return Promise\.all\(\[known, sweepBotProfileSessions\(\)\.catch\(\(\) => undefined\)\]\)/)
})

test('the canonical-chat adoption scan lists with include_hidden', () => {
  // The one session.list consumer that must see the always-hidden rows:
  // findExistingCanonicalChat (the registry lookup) — canonical Bot Chats
  // are born hidden, so a visible-only scan would miss the very row that IS
  // the bot's identity. The lookup rides the bot's own source (requestForBot)
  // and fails CLOSED: a thrown lookup never falls through to minting.
  assert.match(
    source,
    /requestForBot\(bot, 'session\.list', \{\s*profile: backendTargetProfile\(route, name\),\s*title: CANONICAL_CHAT_TITLE,[\s\S]{0,200}?include_hidden: true\s*\}\)\s*\} catch \(error\) \{[\s\S]{0,800}?const rows = res\?\.sessions \?\? \[\]\s*return rows\.find\(row => isCanonicalBotChatHistory\(row\)\)/
  )
})
