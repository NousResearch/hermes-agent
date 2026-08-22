import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

const source = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

function loadFilter() {
  const start = source.indexOf('function botHandle')
  const end = source.indexOf('function slugify')

  assert.ok(start >= 0 && end > start, 'bot identity helper block must remain extractable')

  const context = { botActivitySession: bot => bot?.canonical_session || bot?.last_session || null }
  vm.runInNewContext(
    `${source.slice(start, end)}\nglobalThis.__filterBots = filterBots;`,
    context
  )

  return context.__filterBots
}

const roster = [
  { name: 'agency-audio-designer', title: 'Audio Designer' },
  { name: 'agency-ai-engineer', title: 'AI Engineer' },
  { name: 'default' }
]
const meta = {
  'agency-audio-designer': { title: 'Sound Studio' },
  default: {}
}

test('bot search matches visible display names case-insensitively', () => {
  const filterBots = loadFilter()

  assert.deepEqual(
    filterBots(roster, meta, 'SOUND').map(bot => bot.name),
    ['agency-audio-designer']
  )
})

test('bot search matches profile handles and preserves roster order', () => {
  const filterBots = loadFilter()

  assert.deepEqual(
    filterBots(roster, meta, 'agency-').map(bot => bot.name),
    ['agency-audio-designer', 'agency-ai-engineer']
  )
  assert.deepEqual(
    filterBots(roster, meta, '@hermes').map(bot => bot.name),
    ['default']
  )
  assert.deepEqual(
    filterBots(roster, meta, 'default').map(bot => bot.name),
    ['default']
  )
})

test('bot search also matches roles, message previews, and gateway labels', () => {
  const filterBots = loadFilter()
  const richerRoster = [
    {
      name: 'reviewer',
      connectionLabel: 'Work Studio',
      description: 'Release quality and compliance',
      last_session: { preview: 'Checked the deployment checklist' }
    },
    { name: 'writer', description: 'Editorial support' }
  ]

  assert.equal(filterBots(richerRoster, {}, 'compliance')[0].name, 'reviewer')
  assert.equal(filterBots(richerRoster, {}, 'deployment checklist')[0].name, 'reviewer')
  assert.equal(filterBots(richerRoster, {}, 'work studio')[0].name, 'reviewer')
})

test('blank bot search returns the existing roster reference', () => {
  const filterBots = loadFilter()

  assert.equal(filterBots(roster, meta, '   '), roster)
})

test('Bot pane renders the canonical search field and no-match state', () => {
  assert.match(
    source,
    /const showRosterSearch =\s*gatewayOptions\.length > 1 \|\| rosterItemCount >= BOT_ROSTER_SEARCH_THRESHOLD/
  )
  assert.match(source, /jsx\(SearchField,\s*\{[\s\S]*?placeholder: 'Search bots and group chats…'/)
  assert.match(source, /query \? 'opacity-100!' : 'opacity-50 focus-within:opacity-100'/)
  assert.match(source, /placeholder:text-\(--ui-text-tertiary\)/)
  assert.match(source, /children: jsx\(Codicon, \{ name: 'list-filter' \}\)/)
  assert.match(source, /'aria-live': 'polite'/)
  assert.match(source, /No bots or group chats match “\$\{query\.trim\(\)\}”/)
  assert.match(source, /const initialRosterLoading = !data && !error && roster\.length === 0/)
  assert.match(source, /\(isLoading \|\| initialRosterLoading\) && !roster\.length/)
})
