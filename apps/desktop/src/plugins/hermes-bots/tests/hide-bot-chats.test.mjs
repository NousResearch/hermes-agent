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

test('hideOwnedBotSessions sweeps canonical chats AND room member sessions', async () => {
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
    $botMeta: { get: () => ({ alpha: { chat: 'chat-a' }, beta: { chat: 'chat-b' }, gamma: {} }) },
    $groupChats: {
      get: () => ({
        Core: { sessions: { alpha: 'room-core-a', beta: 'room-core-b' } },
        Quiet: { sessions: { alpha: 'chat-a' } }, // duplicate id — must dedupe
        Legacy: {} // pre-sessions room shape
      })
    }
  }
  const section = source.slice(start, end).concat('\nglobalThis.__h = { hideOwnedBotSessions };\n')
  vm.runInNewContext(section, context, { filename: 'h.js' })
  await context.__h.hideOwnedBotSessions()

  const ids = calls.map(c => c.params.session_id).sort()
  assert.deepEqual(ids, ['chat-a', 'chat-b', 'room-core-a', 'room-core-b'])
  assert.ok(calls.every(c => c.method === 'session.set_hidden' && c.params.hidden === true))
})

test('the Bots session browser lists with include_hidden', () => {
  // The one session.list consumer that must see the always-hidden rows.
  // (Canonical-chat recovery now goes through profiles.list
  // preferred_session_ids, whose resolver already sees hidden rows.)
  assert.match(source, /session\.list', \{ profile: botName, limit: PROFILE_SESSION_LIST_LIMIT, include_hidden: true \}/)
})
