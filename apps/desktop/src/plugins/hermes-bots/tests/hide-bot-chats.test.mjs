import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

// #46: canonical "Bot Chat" sessions can be hidden from the global Sessions
// sidebar via the core generic `hidden` session flag, while staying in the Bots
// roster. The plugin passes hidden:true on session.create when the pref is on,
// and setHideBotChats reconciles existing chats via session.set_hidden.

const source = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

function loadCreate(hidePref) {
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
    $hideBotChats: { get: () => hidePref },
    window: { setTimeout: cb => cb() }
  }
  const section = source.slice(start, end).concat('\nglobalThis.__c = { createCanonicalChat };\n')
  vm.runInNewContext(section, context, { filename: 'c.js' })
  return { create: context.__c.createCanonicalChat, created }
}

test('createCanonicalChat passes hidden:true when the pref is on', async () => {
  const { create, created } = loadCreate(true)
  await create('alpha')
  assert.equal(created.length, 1)
  assert.equal(created[0].hidden, true)
  assert.equal(created[0].title, 'Bot Chat')
})

test('createCanonicalChat omits hidden when the pref is off', async () => {
  const { create, created } = loadCreate(false)
  await create('beta')
  assert.equal(created.length, 1)
  assert.ok(!('hidden' in created[0]), 'hidden should be omitted, not false')
})

test('setHideBotChats persists the pref and reconciles known chats via session.set_hidden', () => {
  // Source-level: the toggle calls session.set_hidden for each known canonical
  // chat id and persists the pref to storage.
  const fn = source.slice(source.indexOf('async function setHideBotChats('), source.indexOf('const avatarFetchInflight'))
  assert.match(fn, /storage\?\.set\?\.\('hide-bot-chats', hidden\)/)
  assert.match(fn, /session\.set_hidden.*session_id: sid, hidden/)
  assert.match(fn, /\.map\(m => m && m\.chat\)/)
})
