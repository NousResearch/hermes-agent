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
    .concat(
      '\nglobalThis.__groups = { groupChatNames, groupLastActivity, groupChatMemberBots, knownGroups, stripPreviewMarkdown, $groupChats };\n'
    )
  vm.runInNewContext(source, context, { filename: 'plugin.js' })
  return context.__groups
}

test('groupChatNames: unions bot-meta groups with room records that carry members or log', () => {
  const { groupChatNames } = load()
  const meta = { researcher: { group: 'Research' }, pm: { group: 'Ops' } }
  const rooms = {
    Research: { log: [], members: [] }, // already known via meta
    Remote: { log: [], members: [{ name: 'spark', remoteSource: true }] },
    Chatty: { log: [{ from: { kind: 'user' }, text: 'hi', at: 5 }] },
    Empty: { log: [], members: [] } // nothing behind it — no row
  }

  const names = groupChatNames(meta, rooms)

  assert.equal(JSON.stringify([...names].sort()), JSON.stringify(['Chatty', 'Ops', 'Remote', 'Research']))
})

test('groupLastActivity: newest room-log timestamp, 0 for silence', () => {
  const { groupLastActivity } = load()

  assert.equal(groupLastActivity({ log: [{ at: 3 }, { at: 9 }] }), 9)
  assert.equal(groupLastActivity({ log: [] }), 0)
  assert.equal(groupLastActivity(undefined), 0)
})

test('groupChatMemberBots: seats local meta members plus stored remote descriptors, preferring live rows', () => {
  const { groupChatMemberBots, $groupChats } = load()
  const roster = [
    { name: 'researcher' },
    { name: 'builder' },
    { name: 'spark', remoteSource: true, connectionId: 'c1', sourceScoped: true }
  ]
  $groupChats.set({
    Research: {
      log: [],
      members: [{ name: 'spark', remoteSource: true, connectionId: 'c1', sourceScoped: true }]
    }
  })

  const members = groupChatMemberBots('Research', roster, {
    researcher: { group: 'Research' },
    builder: { group: 'Ops' }
  })

  assert.equal(JSON.stringify(members.map(m => m.name)), JSON.stringify(['researcher', 'spark']))
  // The LIVE roster row was preferred over the stored descriptor.
  assert.equal(members[1], roster[2])
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

test('stripPreviewMarkdown: flattens bold, quotes, code, and links out of previews', () => {
  const { stripPreviewMarkdown } = load()

  assert.equal(stripPreviewMarkdown('**Plan**: ship the `thing`'), 'Plan: ship the thing')
  assert.equal(stripPreviewMarkdown('> quoted wisdom'), 'quoted wisdom')
  assert.equal(stripPreviewMarkdown('see [the doc](https://x.y/z) now'), 'see the doc now')
  assert.equal(stripPreviewMarkdown('## Heading\nbody'), 'Heading body')
  assert.equal(stripPreviewMarkdown(''), '')
})

test('source contract: the roster is a flat list of bot + group rows and the row menu offers grouping', () => {
  // Flat Discord-style list — the sectioned groupRoster presentation is gone.
  assert.doesNotMatch(pluginSource, /function groupRoster\(/)
  assert.match(pluginSource, /rosterRows\.map\(row =>/)
  assert.match(pluginSource, /function GroupRow\(/)
  assert.match(pluginSource, /onGroup: setGrouping/)
  assert.match(pluginSource, /'Move to group…'/)
})

test('source contract: group rows carry the needs-you badge and open via openGroupChat', () => {
  assert.match(pluginSource, /needsYou: Boolean\(groupNeedsYou\[row\.name\]\)/)
  assert.match(pluginSource, /onOpen: openGroupChat/)
})
