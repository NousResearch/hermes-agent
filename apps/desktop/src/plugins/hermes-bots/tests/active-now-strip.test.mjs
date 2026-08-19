import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

const source = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

// The activeBots helper must stay a self-contained slice between the
// liveness-window constant and the BotRow section so tests can extract it.
function loadActiveBots() {
  const start = source.indexOf('const ACTIVE_WINDOW_S')
  const end = source.indexOf('// ── bot row ─')

  assert.ok(start >= 0 && end > start, 'activeBots block must remain extractable')

  const context = {}
  vm.runInNewContext(
    `${source.slice(start, end)}\nglobalThis.__activeBots = activeBots;`,
    context
  )

  return context.__activeBots
}

// Fixed clock so "inside the window" vs "stale" is deterministic.
const NOW = 1_000_000_000_000

const roster = [
  { name: 'researcher', last_session: { last_active: (NOW / 1000) - 10 } },
  { name: 'scribe', last_session: { last_active: (NOW / 1000) - 400 } },
  { name: 'analyst', last_session: null }
]

test('activeBots includes the gateway-busy selected profile before its first response lands', () => {
  const activeBots = loadActiveBots()
  // analyst has no last_session at all — a busy turn must still show it.
  const names = activeBots(roster, 'analyst', 'busy', NOW).map(bot => bot.name)
  assert.ok(names.includes('analyst'))
})

test('activeBots includes bots whose last message is inside the liveness window', () => {
  const activeBots = loadActiveBots()
  const names = activeBots(roster, 'default', 'open', NOW).map(bot => bot.name)
  assert.ok(names.includes('researcher'))
})

test('activeBots excludes stale activity outside the window', () => {
  const activeBots = loadActiveBots()
  const names = activeBots(roster, 'default', 'open', NOW).map(bot => bot.name)
  assert.ok(!names.includes('scribe'))
})

test('activeBots preserves roster order and never hides other bots', () => {
  const activeBots = loadActiveBots()
  // Mixed window: only researcher qualifies, but the output stays in input
  // order and the full roster object is untouched.
  const active = activeBots(roster, 'default', 'open', NOW)
  assert.deepEqual(active.map(bot => bot.name), ['researcher'])
  assert.equal(roster.length, 3)
})

test('activeBots returns an empty list when nothing is active', () => {
  const activeBots = loadActiveBots()
  const quiet = [
    { name: 'scribe', last_session: { last_active: (NOW / 1000) - 400 } },
    { name: 'analyst', last_session: null }
  ]
  assert.deepEqual(activeBots(quiet, 'default', 'open', NOW), [])
})

test('roster without profiles never throws', () => {
  const activeBots = loadActiveBots()
  assert.equal(activeBots(null, 'default', 'open', NOW).length, 0)
  assert.equal(activeBots([], 'default', 'open', NOW).length, 0)
})

test('ActiveNowStrip renders above the roster, is a live region, and is click-accessible', () => {
  // Strip is placed between the pane header and the search field.
  const headerEnd = source.indexOf("children: 'Bots'")
  const searchField = source.indexOf("placeholder: 'Search bots…'")
  assert.ok(headerEnd >= 0 && searchField > headerEnd)

  const stripStart = source.indexOf('jsx(ActiveNowStrip')
  assert.ok(stripStart > headerEnd && stripStart < searchField, 'strip sits between header and search')

  // Live region announces membership changes politely.
  assert.match(source, /'aria-live': 'polite'/)
  // Chips are real buttons (keyboard/click accessible), reuse BotFace, and
  // open the canonical chat via the same path as roster rows.
  assert.match(source, /jsx\('button', \{\s*type: 'button',\s*title: `Open \$\{label\}'s chat`/)
  // The key rides as jsx()'s third argument — the ONLY form React treats as
  // a list key; a `key:` prop leaves chips unkeyed (index identity).
  assert.match(source, /\}, botRosterKey\(bot\)\)\s*\}\)\s*\]\s*\}\)\s*\}\s*\/\*\* Assign a bot to a group/s)
  assert.match(source, /jsx\(BotFace,\s*\{[\s\S]*?mood: 'work'/)
  assert.match(source, /let pinnedChat = botRosterMeta\(bot, allMeta\)\?\.chat/)
  assert.match(source, /await prepareBotSource\(bot, pinnedChat\)/)
  assert.match(source, /bot\.preferred_session \|\| bot\.last_session/)
})
