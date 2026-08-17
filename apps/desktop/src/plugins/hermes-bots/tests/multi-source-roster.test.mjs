import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

const source = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

function runtime() {
  const atom = value => ({ get: () => value, set: () => undefined })
  const jsx = (type, props = {}) => ({ type, props })
  const context = {
    atom,
    jsx,
    jsxs: jsx,
    useQuery: () => ({}),
    useValue: value => (value?.get ? value.get() : value),
    useState: value => [value, () => undefined],
    document: { getElementById: () => null, createElement: () => ({}), head: { appendChild: () => undefined } },
    host: {
      state: {
        connectionId: { get: () => 'local', listen: () => undefined },
        profile: { get: () => 'ops', listen: () => undefined }
      },
      request: () => undefined
    },
    sdk: new Proxy({}, { get: () => undefined })
  }
  const code = source
    .replace(/^import \* as sdk from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^import\s+\{[\s\S]*?\}\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^import .* from 'react'\r?\n/m, '')
    .replace(/^import .* from 'react\/jsx-runtime'\r?\n/m, '')
    .replace('export default {', 'globalThis.plugin = {')
    .concat(
      '\nglobalThis.__mergeMultiSourceRoster = mergeMultiSourceRoster;\nglobalThis.__botHandle = botHandle;\nglobalThis.__botRosterKey = botRosterKey;\nglobalThis.__botRosterMeta = botRosterMeta;\nglobalThis.__displayName = displayName;\nglobalThis.__filterBots = filterBots;'
    )
  vm.runInNewContext(code, context)
  return context
}

test('merge: no union → local list untouched', () => {
  const { __mergeMultiSourceRoster: merge } = runtime()
  const local = { profiles: [{ name: 'default', last_session: { id: 's1' } }] }

  const out = merge(local, { agents: [] })
  assert.equal(out.profiles.length, 1)
  assert.equal(out.profiles[0].name, 'default')
  assert.equal(out.profiles[0].last_session.id, 's1')
})

test('merge: local rows are annotated, remote rows appended with source tags', () => {
  const { __mergeMultiSourceRoster: merge } = runtime()
  const local = { profiles: [{ name: 'research', last_session: { id: 's1' } }] }
  const union = {
    agents: [
      {
        connectionId: 'local',
        connectionKind: 'local',
        connectionLabel: 'This device',
        profile: 'research',
        handle: 'research-this-device'
      },
      {
        connectionId: 'homelab',
        connectionKind: 'remote',
        connectionLabel: 'Homelab',
        profile: 'research',
        handle: 'research-homelab'
      },
      {
        connectionId: 'homelab',
        connectionKind: 'remote',
        connectionLabel: 'Homelab',
        profile: 'coder',
        handle: 'coder'
      }
    ]
  }

  const out = merge(local, union, 'local')
  assert.equal(out.profiles.length, 3)

  const localRow = out.profiles.find(p => p.name === 'research' && !p.remoteSource)
  // Annotated in place — rich fields survive, handle attached.
  assert.equal(localRow.last_session.id, 's1')
  assert.equal(localRow.handle, 'research-this-device')
  assert.equal(localRow.sourceScoped, true)
  assert.equal(localRow.remoteSource, undefined)

  const remoteRow = out.profiles.find(p => p.name === 'research' && p.remoteSource)
  assert.equal(remoteRow.handle, 'research-homelab')
  assert.equal(remoteRow.connectionId, 'homelab')
  assert.equal(remoteRow.connectionLabel, 'Homelab')

  const coder = out.profiles.find(p => p.name === 'coder')
  assert.equal(coder.remoteSource, true)
  assert.equal(coder.handle, 'coder')
})

test('merge: union-only active profiles are NOT invented as thin rows', () => {
  const { __mergeMultiSourceRoster: merge } = runtime()
  const local = { profiles: [{ name: 'default' }] }
  const union = {
    agents: [
      {
        connectionId: 'local',
        connectionKind: 'local',
        connectionLabel: 'This device',
        profile: 'ghost',
        handle: 'ghost'
      }
    ]
  }

  const out = merge(local, union, 'local')
  assert.equal(out.profiles.length, 1)
  assert.equal(out.profiles[0].name, 'default')
})

test('merge: duplicate source and local identities render once', () => {
  const { __mergeMultiSourceRoster: merge } = runtime()
  const local = {
    profiles: [
      { name: 'default', last_session: { id: 'newest' } },
      { name: 'default', last_session: { id: 'stale' } }
    ]
  }
  const union = {
    agents: [
      { connectionId: 'local', connectionKind: 'local', profile: 'default', handle: 'default-this-device' },
      { connectionId: 'local', connectionKind: 'local', profile: 'default', handle: 'default-this-device' },
      {
        connectionId: 'homelab',
        connectionKind: 'remote',
        connectionLabel: 'Homelab',
        profile: 'default',
        handle: 'default-homelab'
      },
      {
        connectionId: 'homelab',
        connectionKind: 'remote',
        connectionLabel: 'Homelab',
        profile: 'default',
        handle: 'default-homelab'
      }
    ]
  }

  const out = merge(local, union, 'local')

  assert.equal(out.profiles.length, 2)
  assert.equal(out.profiles[0].last_session.id, 'newest')
  assert.equal(out.profiles[0].handle, 'default-this-device')
  assert.equal(out.profiles[1].connectionId, 'homelab')
})

test('merge: rich rows follow the active remote source, not the local source', () => {
  const { __mergeMultiSourceRoster: merge } = runtime()
  const active = { profiles: [{ name: 'default', last_session: { id: 'remote-session' } }] }
  const union = {
    agents: [
      {
        connectionId: 'local',
        connectionKind: 'local',
        connectionLabel: 'This device',
        profile: 'default',
        handle: 'default-this-device'
      },
      {
        connectionId: 'work',
        connectionKind: 'remote',
        connectionLabel: 'Work',
        profile: 'default',
        handle: 'default-work'
      }
    ]
  }

  const out = merge(active, union, 'work')
  const remote = out.profiles.find(p => p.connectionId === 'work')
  const local = out.profiles.find(p => p.connectionId === 'local')

  assert.equal(remote.last_session.id, 'remote-session')
  assert.equal(remote.sourceScoped, true)
  assert.equal(remote.remoteSource, undefined)
  assert.equal(local.remoteSource, true)
  assert.equal(local.sourceScoped, true)
})

test('merge: repeated refreshes stay idempotent and do not mutate gateway rows', () => {
  const { __mergeMultiSourceRoster: merge } = runtime()
  const rich = { name: 'default', last_session: { id: 'remote-session' } }
  const local = { profiles: [rich, rich] }
  const union = {
    agents: [
      {
        connectionId: 'local',
        connectionKind: 'local',
        connectionLabel: 'This device',
        profile: 'default',
        handle: 'default-this-device'
      },
      {
        connectionId: 'work',
        connectionKind: 'remote',
        connectionLabel: 'Work',
        profile: 'default',
        handle: 'default-work'
      },
      {
        connectionId: 'work',
        connectionKind: 'remote',
        connectionLabel: 'Work',
        profile: 'default',
        handle: 'default-work'
      }
    ]
  }

  const once = merge(local, union, 'work')
  const twice = merge(once, union, 'work')
  const identities = twice.profiles.map(row => `${row.connectionId}:${row.name}`)

  assert.equal(identities.join(','), 'work:default,local:default')
  assert.equal(new Set(identities).size, identities.length)
  assert.equal(rich.connectionId, undefined)
})

test('default rows use source identity without borrowing another source title', () => {
  const { __botRosterKey: key, __botRosterMeta: metaFor, __displayName: name } = runtime()
  const remote = {
    name: 'default',
    connectionId: 'personal',
    connectionLabel: 'Personal',
    remoteSource: true,
    sourceScoped: true
  }
  const active = { ...remote, remoteSource: undefined }
  const metadata = { default: { title: 'Active workspace' } }

  assert.equal(metaFor(remote, metadata), null)
  assert.equal(name(remote, metaFor(remote, metadata)), 'Personal')
  assert.equal(key(remote), 'personal::default')

  // The ACTIVE gateway's own default is the user's main agent — annotation
  // (sourceScoped + connection fields) must NOT rename it to a connection
  // label. Titled: the title wins. Untitled: it stays "Hermes". Regression:
  // remote-gateway desktops showed the main agent as an IP-derived label
  // with no shortname (Aug 17 2026 report).
  assert.equal(name(active, metadata.default), 'Active workspace')
  assert.equal(name(active, undefined), 'Hermes')
})

test('botHandle: precomputed multi-source handle wins; default stays hermes', () => {
  const { __botHandle: botHandle } = runtime()

  assert.equal(botHandle('research', { handle: 'research-homelab' }), 'research-homelab')
  assert.equal(botHandle('research', { handle: 'research' }), 'research')
  assert.equal(botHandle('research'), 'research')
  assert.equal(botHandle('default'), 'hermes')
})

test('filterBots: matches the source device name for remote rows', () => {
  const { __filterBots: filterBots } = runtime()
  const roster = [
    { name: 'research' },
    { name: 'research', remoteSource: true, connectionLabel: 'Homelab', handle: 'research-homelab' }
  ]

  const hits = filterBots(roster, {}, 'homelab')
  assert.equal(hits.length, 1)
  assert.equal(hits[0].remoteSource, true)

  // Handle search still narrows to the disambiguated row.
  const byHandle = filterBots(roster, {}, '@research-homelab')
  assert.equal(byHandle.length, 1)
  assert.equal(byHandle[0].handle, 'research-homelab')

  // Bare profile search keeps matching both rows.
  assert.equal(filterBots(roster, {}, 'research').length, 2)
})

test('merge: active-gateway union agents annotate local rows, not duplicates (remote-primary desktop)', () => {
  const { __mergeMultiSourceRoster: merge } = runtime()
  const local = {
    profiles: [
      { name: 'default', last_session: { id: 's-default' } },
      { name: 'dev', last_session: { id: 's-dev' } }
    ]
  }
  const union = {
    primaryConnectionId: '10-244-108-128-9119',
    agents: [
      // The ACTIVE remote gateway itself — same identities as local, must
      // annotate in place rather than append phantom duplicates (#88344).
      { connectionId: '10-244-108-128-9119', connectionKind: 'remote', connectionLabel: '10.244.108.128:9119', profile: 'default', handle: 'default-10-244-108-128-9119' },
      { connectionId: '10-244-108-128-9119', connectionKind: 'remote', connectionLabel: '10.244.108.128:9119', profile: 'dev', handle: 'dev' },
      // A genuinely separate source with a same-named profile keeps its own row.
      { connectionId: 'local', connectionKind: 'local', connectionLabel: 'This device', profile: 'default', handle: 'default-this-device' },
      { connectionId: 'local', connectionKind: 'local', connectionLabel: 'This device', profile: 'agent-mentor', handle: 'agent-mentor' }
    ]
  }

  const out = merge(local, union)
  // 2 annotated local rows + 2 other-source rows = 4, NOT 6 (no phantom copies).
  assert.equal(out.profiles.length, 4)

  const defaultRow = out.profiles.find(p => p.name === 'default' && !p.remoteSource)
  // Annotated with the ACTIVE gateway's handle; rich fields survive.
  assert.equal(defaultRow.last_session.id, 's-default')
  assert.equal(defaultRow.handle, 'default-10-244-108-128-9119')

  // The same-named profile on the local device stays a separate tagged row.
  const localDefault = out.profiles.find(p => p.name === 'default' && p.remoteSource)
  assert.equal(localDefault.handle, 'default-this-device')
  assert.equal(localDefault.connectionId, 'local')

  const mentor = out.profiles.find(p => p.name === 'agent-mentor')
  assert.equal(mentor.remoteSource, true)

  // Exactly the two other-source rows are tagged remote; none from the active gateway.
  assert.equal(out.profiles.filter(p => p.remoteSource).length, 2)
})

test('merge: primary-local desktop keeps single-source behavior (no phantom rows)', () => {
  const { __mergeMultiSourceRoster: merge } = runtime()
  const local = { profiles: [{ name: 'default' }, { name: 'dev' }] }
  const union = {
    primaryConnectionId: 'local',
    agents: [
      { connectionId: 'local', connectionKind: 'local', connectionLabel: 'This device', profile: 'default', handle: 'default-this-device' },
      { connectionId: 'local', connectionKind: 'local', connectionLabel: 'This device', profile: 'dev', handle: 'dev' }
    ]
  }

  const out = merge(local, union)
  assert.equal(out.profiles.length, 2)
  assert.equal(out.profiles.filter(p => p.remoteSource).length, 0)
  assert.equal(out.profiles[0].handle, 'default-this-device')
})

// The LIVE-active-id override: after the user activates a non-primary
// source's agent, profiles.list answers from THAT source — the merge must
// classify against the live id, not the registry primary, or the active
// source's agents duplicate all over again.
test('merge: live-null local window does not treat registry primary as active', () => {
  const { __mergeMultiSourceRoster: merge } = runtime()
  const local = { profiles: [{ name: 'default', last_session: { id: 'this-chat' } }] }
  const union = {
    primaryConnectionId: 'spark',
    agents: [
      {
        connectionId: 'local',
        connectionKind: 'local',
        connectionLabel: 'This device',
        profile: 'default',
        handle: 'default-this-device'
      },
      { connectionId: 'spark', connectionKind: 'ssh', connectionLabel: 'Spark', profile: 'bob', handle: 'bob' },
      { connectionId: 'spark', connectionKind: 'ssh', connectionLabel: 'Spark', profile: 'kai', handle: 'kai' },
      { connectionId: 'spark', connectionKind: 'ssh', connectionLabel: 'Spark', profile: 'rook', handle: 'rook' }
    ]
  }

  // Clicking the local agent leaves host.state.connectionId null while the
  // registry primary stays on Spark. That must not skip Spark bots or invent
  // a second "This device" shadow of default.
  const out = merge(local, union, null)

  assert.equal(out.profiles.filter(p => p.name === 'default').length, 1)
  assert.equal(out.profiles.find(p => p.name === 'default').last_session.id, 'this-chat')
  assert.equal(out.profiles.filter(p => p.remoteSource && p.connectionId === 'local').length, 0)
  assert.equal(
    out.profiles
      .filter(p => p.remoteSource)
      .map(p => p.name)
      .sort()
      .join(','),
    'bob,kai,rook'
  )
})

test('merge: previously seen remotes survive a connect-on-demand empty union', () => {
  const { __mergeMultiSourceRoster: merge } = runtime()
  const previous = [
    { name: 'default', last_session: { id: 'this-chat' } },
    {
      name: 'bob',
      remoteSource: true,
      sourceScoped: true,
      connectionId: 'spark',
      connectionKind: 'ssh',
      connectionLabel: 'Spark',
      handle: 'bob'
    }
  ]
  const local = { profiles: [{ name: 'default', last_session: { id: 'this-chat' } }] }
  const union = {
    primaryConnectionId: 'local',
    agents: [
      {
        connectionId: 'local',
        connectionKind: 'local',
        connectionLabel: 'This device',
        profile: 'default',
        handle: 'default'
      }
    ],
    sources: [{ connectionId: 'spark', kind: 'ssh', error: 'connect-on-demand' }]
  }

  const out = merge(local, union, 'local', previous)
  const bob = out.profiles.find(p => p.name === 'bob' && p.connectionId === 'spark')

  assert.ok(bob)
  assert.equal(bob.remoteSource, true)
  assert.equal(out.profiles.filter(p => p.name === 'default').length, 1)
})

test('displayName: local default stays Hermes; remote default uses the device label', () => {
  const { __displayName: name } = runtime()

  assert.equal(
    name(
      {
        name: 'default',
        sourceScoped: true,
        connectionKind: 'local',
        connectionLabel: 'This device'
      },
      null
    ),
    'Hermes'
  )
  assert.equal(
    name(
      {
        name: 'default',
        sourceScoped: true,
        remoteSource: true,
        connectionKind: 'ssh',
        connectionLabel: 'Spark'
      },
      null
    ),
    'Spark'
  )
})

test('merge: live active id beats primaryConnectionId for active-source matching', () => {
  const { __mergeMultiSourceRoster: merge } = runtime()
  const local = { profiles: [{ name: 'default', last_session: { id: 's1' } }] }
  const union = {
    primaryConnectionId: 'local',
    agents: [
      { connectionId: 'local', connectionKind: 'local', connectionLabel: 'This device', profile: 'default', handle: 'default-this-device' },
      { connectionId: 'vps', connectionKind: 'remote', connectionLabel: 'VPS', profile: 'default', handle: 'default-vps' }
    ]
  }

  // Live gateway = vps: its union row annotates the rich row; the primary
  // (local) row appends as the genuinely-other source.
  const out = merge(local, union, 'vps')
  assert.equal(out.profiles.length, 2)

  const rich = out.profiles.find(p => p.last_session)
  assert.equal(rich.handle, 'default-vps')
  assert.equal(out.profiles.filter(p => p.remoteSource).length, 1)
  assert.equal(out.profiles.find(p => p.remoteSource).connectionId, 'local')
})

test('botRosterKey: same name on two sources yields distinct React keys', () => {
  const { __botRosterKey: botRosterKey } = runtime()

  const legacyRow = botRosterKey({ name: 'default' })
  const remoteRow = botRosterKey({ name: 'default', remoteSource: true, connectionId: 'homelab' })
  const activeRow = botRosterKey({ name: 'default', connectionId: 'vps' })

  assert.notEqual(legacyRow, remoteRow)
  assert.notEqual(activeRow, remoteRow)
  // Single-source desktops (no connection ids anywhere) keep a stable key.
  assert.equal(legacyRow, 'legacy::default')
})
