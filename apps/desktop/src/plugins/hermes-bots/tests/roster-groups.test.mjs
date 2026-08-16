import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

const pluginSource = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

function load() {
  const values = new Map()
  const atom = initial => {
    const slot = { get: () => values.get(slot), set: value => values.set(slot, value) }
    values.set(slot, initial)
    return slot
  }
  const context = {
    atom,
    PALETTE_AREA: 'palette',
    COMPOSER_AREAS: { middleware: 'middleware' },
    document: { getElementById: () => null, createElement: () => ({}), head: { appendChild: () => undefined } },
    host: {
      request: () => Promise.resolve({}),
      state: { profile: { get: () => 'default', listen: () => undefined }, gateway: { listen: () => undefined } }
    }
  }
  const source = pluginSource
    .replace(/^import\s+\*\s+as\s+sdk\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^import\s+\{[\s\S]*?\}\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^const \{ McpTab, ToolsetConfigPanel \} = sdk\r?\n/m, '')
    .replace(/^import .* from 'react'\r?\n/m, '')
    .replace(/^import .* from 'react\/jsx-runtime'\r?\n/m, '')
    .replace('export default {', 'globalThis.plugin = {')
    .concat('\nglobalThis.__groups = { groupRoster, knownGroups };\n')
  vm.runInNewContext(source, context, { filename: 'plugin.js' })
  return context.__groups
}

const ROSTER = [{ name: 'hermes' }, { name: 'researcher' }, { name: 'builder' }, { name: 'pm' }]

test('groupRoster: ungrouped bots lead, groups follow alphabetically, roster order kept inside', () => {
  const { groupRoster } = load()
  const meta = {
    researcher: { group: 'Research' },
    pm: { group: 'Ops' },
    builder: { group: 'Research' }
  }

  const sections = groupRoster(ROSTER, meta)

  // JSON round-trip: vm-realm arrays fail deepEqual on prototype identity.
  assert.equal(
    JSON.stringify(sections.map(s => [s.group, s.bots.map(b => b.name)])),
    JSON.stringify([
      [null, ['hermes']],
      ['Ops', ['pm']],
      ['Research', ['researcher', 'builder']]
    ])
  )
})

test('groupRoster: no groups means one plain section — zero separators', () => {
  const { groupRoster } = load()

  const sections = groupRoster(ROSTER, {})

  assert.equal(sections.length, 1)
  assert.equal(sections[0].group, null)
  assert.equal(sections[0].bots.length, 4)
})

test('groupRoster: blank/whitespace group values count as ungrouped', () => {
  const { groupRoster } = load()

  const sections = groupRoster(ROSTER, { pm: { group: '  ' }, builder: { group: '' }, hermes: { group: null } })

  assert.equal(sections.length, 1)
  assert.equal(sections[0].group, null)
})

test('knownGroups: unique, trimmed, alphabetical', () => {
  const { knownGroups } = load()

  const groups = knownGroups({
    a: { group: 'research' },
    b: { group: 'Ops' },
    c: { group: 'research' },
    d: { group: '' },
    e: {}
  })

  assert.equal(JSON.stringify(groups), JSON.stringify(['Ops', 'research']))
})

test('source contract: the roster render is sectioned and the row menu offers grouping', () => {
  assert.match(pluginSource, /groupRoster\(filteredRoster, allMeta\)\.flatMap/)
  assert.match(pluginSource, /onGroup: setGrouping/)
  assert.match(pluginSource, /'Move to group…'/)
})
