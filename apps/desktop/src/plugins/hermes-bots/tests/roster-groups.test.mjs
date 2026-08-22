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
      '\nglobalThis.__groups = { botGroups, groupMembershipPatch, groupChatNames, groupLastActivity, groupChatMemberBots, durableGroupChatMembers, writeCompleteGroupMembers, hydrateStoredGroupChatRoom, GROUP_CHAT_MEMBERS_SCHEMA, knownGroups, stripPreviewMarkdown, $groupChats };\n'
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

test('groupChatMemberBots: legacy unmarked rooms keep stored remotes plus local-meta members', () => {
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

test('groupChatMemberBots: pre-schema Pi+Mac room keeps the local Mac bot from metadata', () => {
  const { groupChatMemberBots, $groupChats } = load()
  const roster = [
    { name: 'default', handle: 'default-this-device', connectionId: 'local' },
    { name: 'lokay', handle: 'lokay-this-device', connectionId: 'local' },
    { name: 'default', handle: 'default-houseofmarvin', remoteSource: true, connectionId: 'houseofmarvin' }
  ]
  $groupChats.set({
    'Marvin, Lokay': {
      log: [],
      members: [{ name: 'default', handle: 'default-houseofmarvin', remoteSource: true, connectionId: 'houseofmarvin' }]
    }
  })

  const members = groupChatMemberBots('Marvin, Lokay', roster, {
    lokay: { group: 'Marvin, Lokay' }
  })

  assert.equal(
    JSON.stringify(members.map(m => `${m.connectionId}::${m.name}`)),
    JSON.stringify(['local::lokay', 'houseofmarvin::default'])
  )
  assert.equal(members[0], roster[1])
  assert.equal(members[1], roster[2])
})

test('groupChatMemberBots: leaked default metadata cannot seat a phantom twin on an unmarked complete room', () => {
  const { groupChatMemberBots, $groupChats } = load()
  const roster = [
    { name: 'default', handle: 'default-this-device', connectionId: 'local' },
    { name: 'lokay', handle: 'lokay-this-device', connectionId: 'local' },
    { name: 'default', handle: 'default-houseofmarvin', remoteSource: true, connectionId: 'houseofmarvin' }
  ]
  $groupChats.set({
    'Marvin, Lokay': {
      log: [],
      members: [
        { name: 'default', handle: 'default-houseofmarvin', remoteSource: true, connectionId: 'houseofmarvin' },
        { name: 'lokay', handle: 'lokay-this-device', remoteSource: true, connectionId: 'local' }
      ]
    }
  })

  const members = groupChatMemberBots('Marvin, Lokay', roster, {
    default: { group: 'Marvin, Lokay' },
    lokay: { group: 'Marvin, Lokay' }
  })

  assert.equal(
    JSON.stringify(members.map(m => `${m.connectionId}::${m.name}`)),
    JSON.stringify(['houseofmarvin::default', 'local::lokay'])
  )
  assert.equal(members[0], roster[2])
  assert.equal(members[1], roster[1])
})

test('groupChatMemberBots: schema-2 rooms treat the stored list as complete', () => {
  const { groupChatMemberBots, GROUP_CHAT_MEMBERS_SCHEMA, $groupChats } = load()
  const roster = [
    { name: 'default', handle: 'default-this-device', connectionId: 'local' },
    { name: 'lokay', handle: 'lokay-this-device', connectionId: 'local' },
    { name: 'default', handle: 'default-houseofmarvin', remoteSource: true, connectionId: 'houseofmarvin' }
  ]
  $groupChats.set({
    'Marvin, Lokay': {
      log: [],
      membersSchema: GROUP_CHAT_MEMBERS_SCHEMA,
      members: [
        { name: 'default', handle: 'default-houseofmarvin', remoteSource: true, connectionId: 'houseofmarvin' },
        { name: 'lokay', handle: 'lokay-this-device', remoteSource: true, connectionId: 'local' }
      ]
    }
  })

  const members = groupChatMemberBots('Marvin, Lokay', roster, {
    default: { group: 'Marvin, Lokay' },
    lokay: { group: 'Marvin, Lokay' }
  })

  assert.equal(
    JSON.stringify(members.map(m => `${m.connectionId}::${m.name}`)),
    JSON.stringify(['houseofmarvin::default', 'local::lokay'])
  )
})

test('groupChatMemberBots: a complete empty roster cannot be resurrected from legacy metadata', () => {
  const { groupChatMemberBots, GROUP_CHAT_MEMBERS_SCHEMA, $groupChats } = load()
  const roster = [{ name: 'researcher' }, { name: 'builder' }]

  $groupChats.set({
    Research: { log: [], members: [], membersSchema: GROUP_CHAT_MEMBERS_SCHEMA }
  })

  const members = groupChatMemberBots('Research', roster, {
    researcher: { group: 'Research' },
    builder: { group: 'Research' }
  })

  assert.equal(members.length, 0)
})

test('groupChatMemberBots: rooms without durable members still use local bot metadata', () => {
  const { groupChatMemberBots, $groupChats } = load()
  const roster = [{ name: 'researcher' }, { name: 'builder' }]

  $groupChats.set({ Research: { log: [], members: [] } })

  const members = groupChatMemberBots('Research', roster, {
    researcher: { group: 'Research' },
    builder: { groups: ['Ops', 'Research'], group: 'Ops' }
  })

  assert.equal(JSON.stringify(members.map(m => m.name)), JSON.stringify(['researcher', 'builder']))
})

test('writeCompleteGroupMembers: stamps schema 2 onto every selected identity', () => {
  const { writeCompleteGroupMembers, durableGroupChatMembers, GROUP_CHAT_MEMBERS_SCHEMA } = load()
  const selected = [
    { name: 'default', handle: 'default-houseofmarvin', connectionId: 'houseofmarvin', connectionLabel: 'HouseOfMarvin' },
    { name: 'lokay', handle: 'lokay-this-device', connectionId: 'local', connectionLabel: 'This device' }
  ]
  const room = writeCompleteGroupMembers({}, selected)

  assert.equal(room.membersSchema, GROUP_CHAT_MEMBERS_SCHEMA)
  assert.equal(GROUP_CHAT_MEMBERS_SCHEMA, 2)
  assert.deepEqual(JSON.parse(JSON.stringify(room.members)), JSON.parse(JSON.stringify(durableGroupChatMembers(selected))))
})

test('hydrateStoredGroupChatRoom: preserves complete-member schema but leaves legacy rooms unmarked', () => {
  const { hydrateStoredGroupChatRoom, GROUP_CHAT_MEMBERS_SCHEMA } = load()
  const base = {
    log: [{ from: { kind: 'user', name: 'You' }, text: 'hello', at: 1 }],
    members: [{ name: 'lokay', connectionId: 'local' }]
  }

  const current = hydrateStoredGroupChatRoom({ ...base, membersSchema: GROUP_CHAT_MEMBERS_SCHEMA })
  const legacy = hydrateStoredGroupChatRoom(base)

  assert.equal(current.membersSchema, GROUP_CHAT_MEMBERS_SCHEMA)
  assert.equal(Object.prototype.hasOwnProperty.call(legacy, 'membersSchema'), false)
  assert.equal(current.running, false)
  assert.equal(current.epoch, 0)
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
