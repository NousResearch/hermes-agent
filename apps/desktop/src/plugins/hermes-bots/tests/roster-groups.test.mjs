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
      '\nglobalThis.__groups = { botGroups, groupMembershipPatch, groupChatNames, groupLastActivity, groupChatMemberBots, durableGroupChatMembers, knownGroups, stripPreviewMarkdown, $groupChats };\n'
    )
  vm.runInNewContext(source, context, { filename: 'plugin.js' })
  return context.__groups
}

test('botGroups: normalizes canonical and legacy membership without duplicates', () => {
  const { botGroups } = load()

  assert.equal(
    JSON.stringify(
      botGroups({ groups: [' Engineering ', '', 'Research', 'Engineering', null, 7, { name: 'Nope' }], group: 'Operations' })
    ),
    JSON.stringify(['Engineering', 'Research'])
  )
  assert.equal(JSON.stringify(botGroups({ group: 'Legacy' })), JSON.stringify(['Legacy']))
  assert.equal(JSON.stringify(botGroups({ groups: [] })), JSON.stringify([]))
})

test('groupMembershipPatch: toggles one membership and keeps the legacy projection compatible', () => {
  const { groupMembershipPatch } = load()
  const meta = { groups: ['Engineering', 'Research'], group: 'Engineering' }

  assert.equal(
    JSON.stringify(groupMembershipPatch(meta, 'Engineering', true)),
    JSON.stringify({ groups: ['Engineering', 'Research'], group: 'Engineering' })
  )
  assert.equal(
    JSON.stringify(groupMembershipPatch(meta, 'Operations', true)),
    JSON.stringify({ groups: ['Engineering', 'Research', 'Operations'], group: 'Engineering' })
  )
  assert.equal(
    JSON.stringify(groupMembershipPatch(meta, 'Engineering', false)),
    JSON.stringify({ groups: ['Research'], group: 'Research' })
  )
  assert.equal(
    JSON.stringify(groupMembershipPatch({ group: 'Legacy' }, 'Legacy', false)),
    JSON.stringify({ groups: [], group: null })
  )
})

test('groupChatNames: unions bot-meta groups with room records that carry members or log', () => {
  const { groupChatNames } = load()
  const meta = {
    researcher: { group: 'Research' },
    pm: { groups: ['Ops', 'Research'], group: 'Ops' },
    scout: { groups: ['External'], group: 'Stale' }
  }
  const rooms = {
    Research: { log: [], members: [] }, // already known via meta
    Remote: { log: [], members: [{ name: 'spark', remoteSource: true }] },
    Chatty: { log: [{ from: { kind: 'user' }, text: 'hi', at: 5 }] },
    Empty: { log: [], members: [] } // nothing behind it — no row
  }

  const names = groupChatNames(meta, rooms)

  assert.equal(
    JSON.stringify([...names].sort()),
    JSON.stringify(['Chatty', 'External', 'Ops', 'Remote', 'Research'])
  )
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
    builder: { groups: ['Ops', 'Research'], group: 'Ops' }
  })

  assert.equal(JSON.stringify(members.map(m => m.name)), JSON.stringify(['researcher', 'builder', 'spark']))
  // The LIVE roster row was preferred over the stored descriptor.
  assert.equal(members[2], roster[2])
})

test('durableGroupChatMembers: retains active and remote source identities', () => {
  const { durableGroupChatMembers } = load()
  const members = durableGroupChatMembers([
    { name: 'default', handle: 'noah', connectionId: 'noah', connectionKind: 'remote', connectionLabel: 'Noah' },
    {
      name: 'default',
      handle: 'maya',
      connectionId: 'maya',
      connectionKind: 'remote',
      connectionLabel: 'Maya',
      remoteSource: true
    }
  ])

  assert.equal(members.length, 2)
  assert.deepEqual(
    JSON.parse(JSON.stringify(members)),
    [
      {
        name: 'default',
        handle: 'noah',
        connectionId: 'noah',
        connectionKind: 'remote',
        connectionLabel: 'Noah',
        remoteSource: true,
        sourceScoped: true
      },
      {
        name: 'default',
        handle: 'maya',
        connectionId: 'maya',
        connectionKind: 'remote',
        connectionLabel: 'Maya',
        remoteSource: true,
        sourceScoped: true
      }
    ]
  )
})

test('knownGroups: unique, trimmed, alphabetical', () => {
  const { knownGroups } = load()

  const groups = knownGroups({
    a: { group: 'research' },
    b: { groups: ['Ops', 'research'], group: 'Ops' },
    c: { groups: ['Design'] },
    d: { group: '' },
    e: {}
  })

  assert.equal(JSON.stringify(groups), JSON.stringify(['Design', 'Ops', 'research']))
})

test('stripPreviewMarkdown: flattens bold, quotes, code, and links out of previews', () => {
  const { stripPreviewMarkdown } = load()

  assert.equal(stripPreviewMarkdown('**Plan**: ship the `thing`'), 'Plan: ship the thing')
  assert.equal(stripPreviewMarkdown('> quoted wisdom'), 'quoted wisdom')
  assert.equal(stripPreviewMarkdown('see [the doc](https://x.y/z) now'), 'see the doc now')
  assert.equal(stripPreviewMarkdown('## Heading\nbody'), 'Heading body')
  assert.equal(stripPreviewMarkdown(''), '')
})

test('source contract: the roster stays a flat list of bot and group rows', () => {
  // Ordering is deliberately unchanged in this PR; sectioned ordering follows separately.
  assert.doesNotMatch(pluginSource, /function groupRoster\(/)
  assert.match(pluginSource, /rosterRows\.map\(row =>/)
  assert.match(pluginSource, /function GroupRow\(/)
  assert.match(pluginSource, /onGroup: setGrouping/)
})

test('source contract: group rows carry the needs-you badge and open via openGroupChat', () => {
  assert.match(pluginSource, /needsYou: Boolean\(groupNeedsYou\[row\.name\]\)/)
  assert.match(pluginSource, /onOpen: openGroupChat/)
})
