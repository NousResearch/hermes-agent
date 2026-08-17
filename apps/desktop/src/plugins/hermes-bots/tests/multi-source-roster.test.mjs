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
    host: { state: { profile: { get: () => 'ops', listen: () => undefined } }, request: () => undefined },
    sdk: new Proxy({}, { get: () => undefined })
  }
  const code = source
    .replace(/^import \* as sdk from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^import\s+\{[\s\S]*?\}\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^import .* from 'react'\r?\n/m, '')
    .replace(/^import .* from 'react\/jsx-runtime'\r?\n/m, '')
    .replace('export default {', 'globalThis.plugin = {')
    .concat(
      '\nglobalThis.__mergeMultiSourceRoster = mergeMultiSourceRoster;\nglobalThis.__botHandle = botHandle;\nglobalThis.__filterBots = filterBots;'
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
      { connectionId: 'homelab', connectionKind: 'remote', connectionLabel: 'Homelab', profile: 'coder', handle: 'coder' }
    ]
  }

  const out = merge(local, union)
  assert.equal(out.profiles.length, 3)

  const localRow = out.profiles.find(p => p.name === 'research' && !p.remoteSource)
  // Annotated in place — rich fields survive, handle attached.
  assert.equal(localRow.last_session.id, 's1')
  assert.equal(localRow.handle, 'research-this-device')
  assert.equal(localRow.remoteSource, undefined)

  const remoteRow = out.profiles.find(p => p.name === 'research' && p.remoteSource)
  assert.equal(remoteRow.handle, 'research-homelab')
  assert.equal(remoteRow.connectionId, 'homelab')
  assert.equal(remoteRow.connectionLabel, 'Homelab')

  const coder = out.profiles.find(p => p.name === 'coder')
  assert.equal(coder.remoteSource, true)
  assert.equal(coder.handle, 'coder')
})

test('merge: union-only local profiles are NOT invented as thin rows', () => {
  const { __mergeMultiSourceRoster: merge } = runtime()
  const local = { profiles: [{ name: 'default' }] }
  const union = {
    agents: [
      { connectionId: 'local', connectionKind: 'local', connectionLabel: 'This device', profile: 'ghost', handle: 'ghost' }
    ]
  }

  const out = merge(local, union)
  assert.equal(out.profiles.length, 1)
  assert.equal(out.profiles[0].name, 'default')
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
