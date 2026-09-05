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

test('sweepBotProfileSessions hides Bot-Mode-titled rows per roster bot, and only those', async () => {
  // Ownership-based half of the sweep: CLI-born "Agent Inbox" / extra
  // "Bot Chat" rows live in bot profiles but are unknown to $botMeta /
  // $groupChats, so the id-based sweep never reaches them. This sweep lists
  // each roster bot's own sessions and hides rows by exact plumbing title
  // ('Bot Chat', 'Agent Inbox', 'Group: …' prefix) — never a user-titled row.
  const start = source.indexOf('function hideOwnedBotSessions()')
  const end = source.indexOf('/** Fetch server-side avatars', start)
  const calls = []
  const rowsByProfile = {
    alpha: [
      { id: 'a-1', title: 'Bot Chat' },
      { id: 'a-2', title: 'Agent Inbox' },
      { id: 'a-3', title: 'Group: Core' },
      { id: 'a-4', title: 'My real conversation' },
      { id: 'a-5', title: 'Bot Chat notes' } // not an exact title — kept
    ],
    remy: [{ id: 'r-1', title: 'Agent Inbox' }]
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
  await context.__h.sweepBotProfileSessions()

  const lists = calls.filter(c => c.method === 'session.list')
  assert.deepEqual(lists.map(c => c.params.profile).sort(), ['alpha', 'remy'])
  // Visible-rows-only listing keeps the sweep idempotent.
  assert.ok(lists.every(c => !c.params.include_hidden))

  const hidden = calls.filter(c => c.method === 'session.set_hidden')
  assert.deepEqual(
    hidden.map(c => c.params.session_id).sort(),
    ['a-1', 'a-2', 'a-3', 'r-1'],
    'exact plumbing titles only — user-titled rows in bot profiles stay visible'
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
  // the bot's identity.
  assert.match(source, /include_hidden: true\s*\}\)\s*const rows = res\?\.sessions \?\? \[\]\s*return rows\.find\(row => isCanonicalBotChatHistory\(row\)\)/)
})
