/**
 * Tests for electron/connection-registry.ts — the v2 multi-connection
 * registry: label rules (required, unique, @handle disambiguation), input
 * validation, registry normalization from disk, the v1→v2 migration, and the
 * pure upsert/remove/set-primary operations.
 */

import assert from 'node:assert/strict'

import { test } from 'vitest'

import type { ConnectionRegistry } from './connection-registry'
import {
  agentHandle,
  backendScopeKey,
  backendScopePrefix,
  buildAgentRoster,
  connectionDialFieldsChanged,
  connectionIdForLabel,
  labelKey,
  labelSlug,
  LOCAL_CONNECTION_ID,
  mergeConnectionInput,
  migrateV1ToRegistry,
  normalizeConnectionInput,
  normalizeRegistry,
  parseRemoteProfileListing,
  REGISTRY_VERSION,
  rememberSshEnumeration,
  removeConnection,
  resolvedConnectionId,
  resolveRegistryLocalRoute,
  setPrimaryConnection,
  shouldDeferLocalEnumeration,
  shouldRetrySshInventory,
  uniqueLabel,
  updateEligibility,
  upsertConnection
} from './connection-registry'

function emptyRegistry(): ConnectionRegistry {
  return normalizeRegistry(null)
}

// --- labels, slugs, handles ---

test('labelKey is case-insensitive and trimmed', () => {
  assert.equal(labelKey('  Homelab '), 'homelab')
  assert.equal(labelKey('HOMELAB'), labelKey('homelab'))
})

test('labelSlug kebab-cases and never returns empty for non-empty input', () => {
  assert.equal(labelSlug('Work Laptop'), 'work-laptop')
  assert.equal(labelSlug('Spark Box #2'), 'spark-box-2')
  assert.equal(labelSlug('!!!'), 'connection')
})

test('resolvedConnectionId identifies local and migrated remote descriptors', () => {
  const registry = migrateV1ToRegistry({
    mode: 'local',
    profiles: {
      personal: { mode: 'remote', url: 'https://personal.example:9443/', authMode: 'token' },
      work: { mode: 'ssh', host: 'work-host', user: 'root' }
    }
  })

  const personal = registry.connections.find(connection => connection.kind === 'remote')
  const work = registry.connections.find(connection => connection.kind === 'ssh')

  assert.equal(resolvedConnectionId(registry, { mode: 'local' }), LOCAL_CONNECTION_ID)
  assert.equal(
    resolvedConnectionId(registry, {
      baseUrl: 'https://personal.example:9443',
      mode: 'remote',
      remoteKind: 'url'
    }),
    personal?.id
  )
  assert.equal(
    resolvedConnectionId(registry, {
      baseUrl: 'http://127.0.0.1:49152',
      mode: 'remote',
      remoteHost: 'root@work-host',
      remoteKind: 'ssh'
    }),
    work?.id
  )
})

test('resolvedConnectionId does not guess an unregistered remote', () => {
  assert.equal(
    resolvedConnectionId(emptyRegistry(), {
      baseUrl: 'https://unknown.example',
      mode: 'remote',
      remoteKind: 'url'
    }),
    null
  )
})

test('agentHandle bare when unique, @name-device shape when duplicated', () => {
  assert.equal(agentHandle('research', 'Homelab', false), 'research')
  assert.equal(agentHandle('research', 'Homelab', true), 'research-homelab')
  assert.equal(agentHandle('research', 'Work Laptop', true), 'research-work-laptop')
  assert.equal(agentHandle('', 'Homelab', false), 'default')
})

test('connectionIdForLabel suffixes on collision and never mints "local"', () => {
  assert.equal(connectionIdForLabel('Homelab', []), 'homelab')
  assert.equal(connectionIdForLabel('Homelab', ['homelab']), 'homelab-2')
  assert.equal(connectionIdForLabel('Homelab', ['homelab', 'homelab-2']), 'homelab-3')
  assert.equal(connectionIdForLabel('Local', []), 'local-2')
})

test('uniqueLabel counts up (never "X 2 2") and clamps long candidates', () => {
  assert.equal(uniqueLabel('Homelab', []), 'Homelab')
  assert.equal(uniqueLabel('Homelab', ['Homelab']), 'Homelab 2')
  assert.equal(uniqueLabel('Homelab', ['Homelab', 'Homelab 2']), 'Homelab 3')
  // Case-insensitive collision detection.
  assert.equal(uniqueLabel('homelab', ['HOMELAB']), 'homelab 2')

  const long = 'x'.repeat(300)
  assert.ok(uniqueLabel(long, []).length <= 64)
  assert.ok(uniqueLabel(long, [uniqueLabel(long, [])]).length <= 64)
})

// --- backendScopeKey (composite pool keys) ---

// The electron and @hermes/shared implementations MUST stay byte-identical —
// the renderer keys its socket registry with the shared copy while the main
// process keys the backend pool with this one. This contract test is the
// enforcement (see the NOTE on backendScopeKey).
test('backendScopeKey: electron and shared implementations agree everywhere', async () => {
  // Non-literal specifier on purpose: tsconfig.electron.json's project
  // boundary excludes apps/shared sources, but vitest resolves the workspace
  // package fine at runtime — which is exactly what this test needs.
  const shared = (await import(String('@hermes/shared'))) as {
    backendScopeKey: typeof backendScopeKey
    backendScopePrefix: typeof backendScopePrefix
    LOCAL_CONNECTION_ID: string
  }

  const cases: [null | string | undefined, null | string | undefined][] = [
    [null, null],
    [undefined, undefined],
    ['', ''],
    ['local', 'research'],
    ['homelab', 'research'],
    ['homelab', ''],
    ['  homelab  ', '  research  '],
    ['spark-2', 'default']
  ]

  for (const [conn, profile] of cases) {
    assert.equal(backendScopeKey(conn, profile), shared.backendScopeKey(conn, profile))
  }

  assert.equal(backendScopePrefix('homelab'), shared.backendScopePrefix('homelab'))
  assert.equal(LOCAL_CONNECTION_ID, shared.LOCAL_CONNECTION_ID)
})

test('backendScopeKey: local/empty connection keeps the bare profile key', () => {
  assert.equal(backendScopeKey(null, 'research'), 'research')
  assert.equal(backendScopeKey('', 'research'), 'research')
  assert.equal(backendScopeKey(LOCAL_CONNECTION_ID, 'research'), 'research')
  assert.equal(backendScopeKey('local', ''), 'default')
  assert.equal(backendScopeKey(undefined, undefined), 'default')
})

test('backendScopeKey: non-local connections get an unambiguous composite', () => {
  assert.equal(backendScopeKey('homelab', 'research'), 'conn:homelab::research')
  assert.equal(backendScopeKey('homelab', ''), 'conn:homelab::default')
  // Composite keys can never collide with a plain profile name, and the
  // prefix helper matches exactly the keys the connection owns.
  assert.ok(backendScopeKey('homelab', 'research').startsWith(backendScopePrefix('homelab')))
  assert.ok(!backendScopeKey('homelab-2', 'research').startsWith(backendScopePrefix('homelab')))
  assert.ok(!'research'.startsWith(backendScopePrefix('homelab')))
})

// --- resolveRegistryLocalRoute (registry 'local' entry vs the v1 route) ---

test('registry local route: delegates to the legacy path when v1 is local (single-source users byte-identical)', () => {
  assert.deepEqual(resolveRegistryLocalRoute('research', {}), { delegate: true, poolKey: 'research' })
  assert.deepEqual(resolveRegistryLocalRoute('', {}), { delegate: true, poolKey: 'default' })
  assert.deepEqual(resolveRegistryLocalRoute(null, { globalRemote: false }), { delegate: true, poolKey: 'default' })
})

test('registry local route: v1 REMOTE global mode forces a genuinely-local backend (migration scenario)', () => {
  // The migration keeps the mandatory 'local' entry AND makes the v1 remote
  // the registry primary. If 'local' delegated to the v1 route here, the
  // roster's "This device" rows would enumerate + dial the REMOTE primary —
  // every profile duplicated and local agents talking to the remote box.
  const route = resolveRegistryLocalRoute('default', { globalRemote: true })

  assert.equal(route.delegate, false)
  // The forced-local child must NOT pool under the bare profile key: that
  // slot is where the v1 route caches the REMOTE descriptor. The composite
  // form is prefix-owned by the local connection and collision-free.
  assert.equal(route.poolKey, 'conn:local::default')
  assert.ok(route.poolKey.startsWith(backendScopePrefix(LOCAL_CONNECTION_ID)))
  assert.notEqual(route.poolKey, backendScopeKey(LOCAL_CONNECTION_ID, 'default'))
})

test('registry local route: a per-profile remote override also forces local', () => {
  const route = resolveRegistryLocalRoute('research', { profileRemoteOverride: true })

  assert.deepEqual(route, { delegate: false, poolKey: 'conn:local::research' })
})

// --- shouldDeferLocalEnumeration (roster's connect-on-demand for 'local') ---

test('local enumeration: delegate route (local-primary desktop) always enumerates', () => {
  const route = resolveRegistryLocalRoute('default', {})

  assert.equal(shouldDeferLocalEnumeration(route, []), false)
  assert.equal(shouldDeferLocalEnumeration(route, ['conn:local::default']), false)
})

test('local enumeration: forced-local route defers until a local child exists (remote-primary desktop)', () => {
  // Remote-gateway-only desktops: enumerating "This device" here would SPAWN
  // a local backend the user never asked for — a phantom `default` agent
  // that duplicates their real one and forces -device handles onto it.
  const route = resolveRegistryLocalRoute('default', { globalRemote: true })

  assert.equal(shouldDeferLocalEnumeration(route, []), true)
  // The v1 remote descriptor cached at the BARE profile key is not a local child.
  assert.equal(shouldDeferLocalEnumeration(route, ['default', 'research']), true)
  // Once the user has genuinely opened a forced-local child, it enumerates.
  assert.equal(shouldDeferLocalEnumeration(route, ['conn:local::default']), false)
})

// --- buildAgentRoster (union roster + @name-device rule) ---

test('roster: unique profiles keep bare handles; duplicates get @name-device', () => {
  const local = { id: 'local', kind: 'local' as const, label: 'This device' }
  const homelab = { id: 'homelab', kind: 'remote' as const, label: 'Homelab', url: 'http://h:1' }

  const roster = buildAgentRoster([
    { connection: local, profiles: ['default', 'research'] },
    { connection: homelab, profiles: ['research', 'coder'] }
  ])

  const byKey = new Map(roster.map(a => [`${a.connectionId}/${a.profile}`, a.handle]))

  // research exists on both sources → both disambiguate.
  assert.equal(byKey.get('local/research'), 'research-this-device')
  assert.equal(byKey.get('homelab/research'), 'research-homelab')
  // default and coder are unique → bare names.
  assert.equal(byKey.get('local/default'), 'default')
  assert.equal(byKey.get('homelab/coder'), 'coder')
  assert.equal(roster.length, 4)
})

test('rememberSshEnumeration: live list wins, cache then seed default', () => {
  assert.deepEqual(rememberSshEnumeration({ profiles: ['bob', 'kai'] }, ['stale'], 'ssh'), {
    profiles: ['bob', 'kai']
  })
  assert.deepEqual(
    rememberSshEnumeration({ profiles: null, error: 'connect-on-demand' }, ['bob', 'kai', 'rook'], 'ssh'),
    { profiles: ['bob', 'kai', 'rook'], error: 'connect-on-demand' }
  )
  assert.deepEqual(rememberSshEnumeration({ profiles: null, error: 'connect-on-demand' }, null, 'ssh'), {
    profiles: ['default'],
    error: 'connect-on-demand'
  })
  assert.deepEqual(rememberSshEnumeration({ profiles: null, error: 'connect-on-demand' }, null, 'remote'), {
    profiles: null,
    error: 'connect-on-demand'
  })
})

test('shouldRetrySshInventory: first try, cooldown, then retry; cache never retries', () => {
  assert.equal(shouldRetrySshInventory(false, null, 1_000), true)
  assert.equal(shouldRetrySshInventory(false, 1_000, 30_000, 60_000), false)
  assert.equal(shouldRetrySshInventory(false, 1_000, 61_000, 60_000), true)
  assert.equal(shouldRetrySshInventory(true, 1_000, 120_000, 60_000), false)
})

test('parseRemoteProfileListing: Mini/Spark dirs become roster names and drop rollbacks', () => {
  const listed = parseRemoteProfileListing(
    ['bob', 'dixie', 'goose', 'rambo', 'bob.rollback-old', '.hidden', '', 'not a name'].join('\n')
  )

  assert.deepEqual(listed, ['default', 'bob', 'dixie', 'goose', 'rambo'])
})

test('parseRemoteProfileListing: empty listing is still the default agent', () => {
  assert.deepEqual(parseRemoteProfileListing(''), ['default'])
})

test('roster: unreachable sources contribute no rows and cannot fake duplicates', () => {
  const local = { id: 'local', kind: 'local' as const, label: 'This device' }
  const dead = { id: 'dead', kind: 'remote' as const, label: 'Dead box', url: 'http://d:1' }

  const roster = buildAgentRoster([
    { connection: local, profiles: ['research'] },
    { connection: dead, profiles: null, error: 'unreachable' }
  ])

  assert.equal(roster.length, 1)
  // Only one live source has research → bare handle, no phantom duplicate.
  assert.equal(roster[0].handle, 'research')
})

test('roster: duplicate profiles from one connection remain one routable agent', () => {
  const local = { id: 'local', kind: 'local' as const, label: 'This device' }
  const homelab = { id: 'homelab', kind: 'remote' as const, label: 'Homelab', url: 'http://h:1' }

  const roster = buildAgentRoster([
    { connection: local, profiles: ['default', 'research', 'default'] },
    // A duplicate registry enumeration must not make local/research a second
    // bot identity either.
    { connection: local, profiles: ['research'] },
    { connection: homelab, profiles: ['research', 'research'] }
  ])

  assert.deepEqual(
    roster.map(agent => `${agent.connectionId}/${agent.profile}`),
    ['local/default', 'local/research', 'homelab/research']
  )
  assert.equal(
    roster.find(agent => agent.connectionId === 'local' && agent.profile === 'research')?.handle,
    'research-this-device'
  )
  assert.equal(
    roster.find(agent => agent.connectionId === 'homelab' && agent.profile === 'research')?.handle,
    'research-homelab'
  )
})

// --- buildAgentRoster: same-backend (install_id) collapse ---

test('roster: two connections with the same install_id collapse to one row per profile', () => {
  const hostname = { id: 'spark', kind: 'remote' as const, label: 'Spark', url: 'http://spark:8642' }
  const tailscale = { id: 'spark-ts', kind: 'remote' as const, label: 'Spark TS', url: 'http://100.1.2.3:8642' }

  const roster = buildAgentRoster([
    { connection: hostname, profiles: ['default', 'research'], installId: 'aaa' },
    { connection: tailscale, profiles: ['default', 'research'], installId: 'aaa' }
  ])

  // One row per (install, profile) — no duplicate bots for the same box.
  assert.deepEqual(roster.map(agent => `${agent.connectionId}/${agent.profile}`).sort(), [
    'spark/default',
    'spark/research'
  ])
  // Handle disambiguation runs AFTER the collapse: no more suffixed names.
  assert.deepEqual(roster.map(agent => agent.handle).sort(), ['default', 'research'])
})

test('roster: collapse prefers the active (primary) connection', () => {
  const hostname = { id: 'spark', kind: 'remote' as const, label: 'Spark', url: 'http://spark:8642' }
  const tailscale = { id: 'spark-ts', kind: 'remote' as const, label: 'Spark TS', url: 'http://100.1.2.3:8642' }

  const roster = buildAgentRoster(
    [
      { connection: hostname, profiles: ['default'], installId: 'aaa' },
      { connection: tailscale, profiles: ['default'], installId: 'aaa' }
    ],
    { primaryConnectionId: 'spark-ts' }
  )

  assert.equal(roster.length, 1)
  assert.equal(roster[0].connectionId, 'spark-ts')
})

test('roster: collapse pick order is local > ssh > remote > cloud, then registration order', () => {
  const local = { id: 'local', kind: 'local' as const, label: 'This device' }
  const remote = { id: 'loop', kind: 'remote' as const, label: 'Loopback', url: 'http://127.0.0.1:8642' }
  const cloud = { id: 'cl', kind: 'cloud' as const, label: 'Cloud twin', url: 'http://cl:1' }
  const ssh = { id: 'tun', kind: 'ssh' as const, label: 'Tunnel', host: 'box' }

  // Same box registered four ways; primary is NOT one of them (unset).
  const roster = buildAgentRoster([
    { connection: cloud, profiles: ['default'], installId: 'aaa' },
    { connection: remote, profiles: ['default'], installId: 'aaa' },
    { connection: ssh, profiles: ['default'], installId: 'aaa' },
    { connection: local, profiles: ['default'], installId: 'aaa' }
  ])

  assert.equal(roster.length, 1)
  assert.equal(roster[0].connectionId, 'local')

  // Without the local candidate, ssh wins over remote/cloud.
  const noLocal = buildAgentRoster([
    { connection: cloud, profiles: ['default'], installId: 'aaa' },
    { connection: remote, profiles: ['default'], installId: 'aaa' },
    { connection: ssh, profiles: ['default'], installId: 'aaa' }
  ])

  assert.equal(noLocal[0].connectionId, 'tun')

  // Same kind → earliest-registered (enumeration order) wins.
  const twin = { id: 'loop2', kind: 'remote' as const, label: 'Loopback 2', url: 'http://[::1]:8642' }

  const sameKind = buildAgentRoster([
    { connection: remote, profiles: ['default'], installId: 'aaa' },
    { connection: twin, profiles: ['default'], installId: 'aaa' }
  ])

  assert.equal(sameKind[0].connectionId, 'loop')
})

test('roster: missing install_id bypasses the collapse (older backends keep current behavior)', () => {
  const hostname = { id: 'spark', kind: 'remote' as const, label: 'Spark', url: 'http://spark:8642' }
  const tailscale = { id: 'spark-ts', kind: 'remote' as const, label: 'Spark TS', url: 'http://100.1.2.3:8642' }

  // Neither reports an id → both rows survive, handles disambiguate.
  const roster = buildAgentRoster([
    { connection: hostname, profiles: ['default'] },
    { connection: tailscale, profiles: ['default'] }
  ])

  assert.equal(roster.length, 2)
  assert.deepEqual(roster.map(a => a.handle).sort(), ['default-spark', 'default-spark-ts'])

  // One id + one missing must NOT collapse either (undefined never matches).
  const mixed = buildAgentRoster([
    { connection: hostname, profiles: ['default'], installId: 'aaa' },
    { connection: tailscale, profiles: ['default'] }
  ])

  assert.equal(mixed.length, 2)
})

test('roster: different install_ids stay separate rows with disambiguated handles', () => {
  const spark = { id: 'spark', kind: 'remote' as const, label: 'Spark', url: 'http://spark:8642' }
  const mini = { id: 'mini', kind: 'remote' as const, label: 'Mini', url: 'http://mini:8642' }

  const roster = buildAgentRoster([
    { connection: spark, profiles: ['default'], installId: 'aaa' },
    { connection: mini, profiles: ['default'], installId: 'bbb' }
  ])

  assert.equal(roster.length, 2)
  assert.deepEqual(roster.map(a => a.handle).sort(), ['default-mini', 'default-spark'])
})

test('roster: collapse also folds a third same-box connection from a per-profile v1 override import', () => {
  // The reporter's "profile with cron appears as another duplicate": the v1
  // migration imports per-profile override blocks as EXTRA connections, so a
  // cron profile pinned to the same box via a third URL becomes a third
  // registry entry. Same install_id → still one row per profile.
  const hostname = { id: 'spark', kind: 'remote' as const, label: 'Spark', url: 'http://spark:8642' }
  const tailscale = { id: 'spark-ts', kind: 'remote' as const, label: 'Spark TS', url: 'http://100.1.2.3:8642' }
  const override = { id: 'spark-lan', kind: 'remote' as const, label: 'Spark LAN', url: 'http://192.168.1.5:8642' }

  const roster = buildAgentRoster([
    { connection: hostname, profiles: ['default', 'cron-bot'], installId: 'aaa' },
    { connection: tailscale, profiles: ['default', 'cron-bot'], installId: 'aaa' },
    { connection: override, profiles: ['default', 'cron-bot'], installId: 'aaa' }
  ])

  assert.deepEqual(roster.map(agent => `${agent.profile}:${agent.handle}`).sort(), [
    'cron-bot:cron-bot',
    'default:default'
  ])
})

// --- updateEligibility ---

test('update fan-out: cloud is platform-managed, everything else eligible', () => {
  assert.deepEqual(updateEligibility({ id: 'c', kind: 'cloud', label: 'Cloud' }), {
    eligible: false,
    reason: 'cloud-managed'
  })
  assert.equal(updateEligibility({ id: 'local', kind: 'local', label: 'x' }).eligible, true)
  assert.equal(updateEligibility({ id: 'r', kind: 'remote', label: 'x' }).eligible, true)
  assert.equal(updateEligibility({ id: 's', kind: 'ssh', label: 'x' }).eligible, true)
})

// --- normalizeConnectionInput ---

test('save rejects the reserved "local" id on non-local kinds', () => {
  assert.throws(
    () =>
      normalizeConnectionInput({ id: 'local', kind: 'remote', label: 'Sneaky', url: 'http://x:1' }, emptyRegistry()),
    /reserved/
  )
})

test('token only persists on token-auth remotes; oauth/cloud drop it', () => {
  const registry = emptyRegistry()

  const tokenAuth = normalizeConnectionInput(
    { kind: 'remote', label: 'A', url: 'http://a:1', authMode: 'token', token: { enc: 'x' } },
    registry
  )

  assert.deepEqual(tokenAuth.token, { enc: 'x' })

  const oauth = normalizeConnectionInput(
    { kind: 'remote', label: 'B', url: 'http://b:1', authMode: 'oauth', token: { enc: 'x' } },
    registry
  )

  assert.equal(oauth.token, undefined)

  const cloud = normalizeConnectionInput(
    { kind: 'cloud', label: 'C', url: 'https://c.hermes.cloud', authMode: 'oauth', token: { enc: 'x' } },
    registry
  )

  assert.equal(cloud.token, undefined)
})

// --- mergeConnectionInput (edit inheritance) ---

test('merge preserves fields the editor does not carry (org, ssh extras)', () => {
  const cloud = {
    authMode: 'oauth' as const,
    id: 'c',
    kind: 'cloud' as const,
    label: 'Cloud',
    org: 'nous',
    url: 'https://a.cloud'
  }

  const renamed = mergeConnectionInput({ id: 'c', kind: 'cloud', label: 'Renamed', url: 'https://a.cloud' }, cloud)

  assert.equal(renamed.org, 'nous')

  const ssh = {
    host: 'homelab.lan',
    id: 's',
    keyPath: '/k/id',
    kind: 'ssh' as const,
    label: 'Box',
    port: 2222,
    remoteHermesPath: '/opt/hermes',
    remoteProfile: 'research',
    user: 'k'
  }

  const labelOnly = mergeConnectionInput({ id: 's', kind: 'ssh', label: 'Renamed box' }, ssh)

  assert.equal(labelOnly.remoteHermesPath, '/opt/hermes')
  assert.equal(labelOnly.remoteProfile, 'research')
  assert.equal(labelOnly.host, 'homelab.lan')
  assert.equal(labelOnly.user, 'k')
  assert.equal(labelOnly.port, 2222)
})

test('merge: a supplied ssh host string beats stored user/port', () => {
  const ssh = { host: 'spark1', id: 's', kind: 'ssh' as const, label: 'Spark', port: 2222, user: 'tek' }
  const merged = mergeConnectionInput({ host: 'admin@newbox:2200', id: 's', kind: 'ssh', label: 'Spark' }, ssh)

  // Stored user/port must NOT ride along — the host string is authoritative.
  assert.equal(merged.user, undefined)
  assert.equal(merged.port, undefined)

  const entry = normalizeConnectionInput(merged, emptyRegistry())

  assert.equal(entry.host, 'newbox')
  assert.equal(entry.user, 'admin')
  assert.equal(entry.port, 2200)
})

test('save rejects a missing label with a device-name message', () => {
  assert.throws(
    () => normalizeConnectionInput({ kind: 'remote', label: '  ', url: 'http://10.0.0.5:9119' }, emptyRegistry()),
    /device name/
  )
})

test('save rejects a duplicate label case-insensitively', () => {
  let registry = emptyRegistry()
  registry = upsertConnection(
    registry,
    normalizeConnectionInput({ kind: 'remote', label: 'Homelab', url: 'http://10.0.0.5:9119' }, registry)
  )

  assert.throws(
    () => normalizeConnectionInput({ kind: 'remote', label: ' homelab ', url: 'http://10.0.0.9:9119' }, registry),
    /must be unique/
  )
})

test('editing an entry does not collide with its own label', () => {
  let registry = emptyRegistry()
  const entry = normalizeConnectionInput({ kind: 'remote', label: 'Homelab', url: 'http://10.0.0.5:9119' }, registry)
  registry = upsertConnection(registry, entry)

  const edited = normalizeConnectionInput(
    { id: entry.id, kind: 'remote', label: 'Homelab', url: 'http://10.0.0.6:9119' },
    registry
  )

  assert.equal(edited.id, entry.id)
  assert.equal(edited.url, 'http://10.0.0.6:9119')
})

test('duplicate gateway URLs are rejected across remote and cloud kinds', () => {
  let registry = emptyRegistry()
  registry = upsertConnection(
    registry,
    normalizeConnectionInput({ kind: 'remote', label: 'Homelab', url: 'http://10.0.0.5:9119' }, registry)
  )

  // Same URL modulo trailing slash → dupe, even as a different kind.
  assert.throws(
    () => normalizeConnectionInput({ kind: 'remote', label: 'Twin', url: 'http://10.0.0.5:9119/' }, registry),
    /already exists/
  )
  assert.throws(
    () => normalizeConnectionInput({ kind: 'cloud', label: 'Cloud twin', url: 'http://10.0.0.5:9119' }, registry),
    /already exists/
  )
  // Editing the entry itself keeps its own URL without self-colliding.
  const existing = registry.connections.find(c => c.kind === 'remote')!

  const edited = normalizeConnectionInput(
    { id: existing.id, kind: 'remote', label: 'Homelab', url: 'http://10.0.0.5:9119' },
    registry
  )

  assert.equal(edited.id, existing.id)
})

test('duplicate ssh targets are rejected on user@host:port + remote profile', () => {
  let registry = emptyRegistry()
  registry = upsertConnection(
    registry,
    normalizeConnectionInput({ kind: 'ssh', label: 'Box', host: 'alice@box:22', remoteProfile: 'work' }, registry)
  )

  assert.throws(
    () =>
      normalizeConnectionInput(
        { kind: 'ssh', label: 'Box twin', host: 'alice@box:22', remoteProfile: 'work' },
        registry
      ),
    /already exists/
  )

  // A different remote profile on the same host is a distinct agent source.
  const otherProfile = normalizeConnectionInput(
    { kind: 'ssh', label: 'Box other', host: 'alice@box:22', remoteProfile: 'other' },
    registry
  )

  assert.equal(otherProfile.kind, 'ssh')
})

test('remote input normalizes URL and auth mode; cloud keeps org', () => {
  const registry = emptyRegistry()

  const remote = normalizeConnectionInput(
    { kind: 'remote', label: 'LAN box', url: '10.0.0.5:9119', authMode: 'weird' },
    registry
  )

  assert.equal(remote.url, 'http://10.0.0.5:9119')
  assert.equal(remote.authMode, 'token')

  const cloud = normalizeConnectionInput(
    { kind: 'cloud', label: 'Cloud', url: 'https://foo.hermes.cloud', authMode: 'oauth', org: 'nous' },
    registry
  )

  assert.equal(cloud.kind, 'cloud')
  assert.equal(cloud.org, 'nous')
  assert.equal(cloud.authMode, 'oauth')
})

test('ssh input requires a host; local input only carries the label', () => {
  const registry = emptyRegistry()

  assert.throws(() => normalizeConnectionInput({ kind: 'ssh', label: 'Spark', host: ' ' }, registry), /host/)

  const ssh = normalizeConnectionInput({ kind: 'ssh', label: 'Spark', host: 'tek@spark1:2222' }, registry)

  assert.equal(ssh.host, 'spark1')
  assert.equal(ssh.user, 'tek')
  assert.equal(ssh.port, 2222)

  const local = normalizeConnectionInput({ kind: 'local', label: 'My MacBook' }, registry)

  assert.equal(local.id, LOCAL_CONNECTION_ID)
  assert.deepEqual(Object.keys(local).sort(), ['id', 'kind', 'label'])
})

// --- normalizeRegistry ---

test('normalizeRegistry degrades junk to a local-only registry', () => {
  for (const junk of [null, undefined, 42, 'nope', { connections: 'zzz' }, { version: 99 }]) {
    const registry = normalizeRegistry(junk)

    assert.equal(registry.version, REGISTRY_VERSION)
    assert.equal(registry.primary, LOCAL_CONNECTION_ID)
    assert.equal(registry.connections.length, 1)
    assert.equal(registry.connections[0].kind, 'local')
  }
})

test('normalizeRegistry guarantees local, dedupes labels, fixes dangling primary', () => {
  const registry = normalizeRegistry({
    version: 2,
    primary: 'ghost',
    connections: [
      { id: 'a', kind: 'remote', label: 'Homelab', url: 'http://10.0.0.5:9119' },
      { id: 'b', kind: 'remote', label: 'homelab', url: 'http://10.0.0.6:9119' },
      { id: 'c', kind: 'remote', label: 'No URL entry' },
      { kind: 'nonsense', label: 'x' }
    ]
  })

  assert.equal(registry.primary, LOCAL_CONNECTION_ID)
  assert.ok(registry.connections.some(c => c.kind === 'local'))

  const labels = registry.connections.map(c => labelKey(c.label))

  assert.equal(new Set(labels).size, labels.length)
  // The url-less remote entry is dropped, the junk kind is dropped.
  assert.equal(registry.connections.filter(c => c.kind === 'remote').length, 2)
})

test('normalizeRegistry round-trips a valid registry unchanged in shape', () => {
  const input = {
    version: 2,
    primary: 'homelab',
    connections: [
      { id: 'local', kind: 'local', label: 'This device' },
      {
        id: 'homelab',
        kind: 'remote',
        label: 'Homelab',
        url: 'http://10.0.0.5:9119',
        authMode: 'token',
        token: { v: 1 }
      },
      {
        id: 'cloud-1',
        kind: 'cloud',
        label: 'Hermes Cloud',
        url: 'https://a.hermes.cloud',
        authMode: 'oauth',
        org: 'nous'
      },
      { id: 'spark', kind: 'ssh', label: 'Spark', host: 'spark1', user: 'tek', port: 2222 }
    ]
  }

  const registry = normalizeRegistry(input)

  assert.equal(registry.primary, 'homelab')
  assert.equal(registry.connections.length, 4)
  assert.deepEqual(
    registry.connections.map(c => c.id),
    ['local', 'homelab', 'cloud-1', 'spark']
  )
  assert.deepEqual(registry.connections[1].token, { v: 1 })
  assert.equal(registry.connections[3].port, 2222)
})

// --- v1 → v2 migration ---

test('migrate: v1 local-only config → local-only registry', () => {
  const registry = migrateV1ToRegistry({ mode: 'local', remote: {}, profiles: {} })

  assert.equal(registry.primary, LOCAL_CONNECTION_ID)
  assert.equal(registry.connections.length, 1)
})

test('migrate: v1 global remote becomes a labeled entry and the primary', () => {
  const registry = migrateV1ToRegistry({
    mode: 'remote',
    remote: { url: 'http://homelab.lan:9119', authMode: 'token', token: { enc: 'x' } }
  })

  const remote = registry.connections.find(c => c.kind === 'remote')

  assert.ok(remote)
  assert.equal(registry.primary, remote.id)
  assert.equal(remote.label, 'homelab.lan:9119')
  assert.deepEqual(remote.token, { enc: 'x' })
})

test('migrate: v1 cloud keeps cloud provenance + org', () => {
  const registry = migrateV1ToRegistry({
    mode: 'cloud',
    remote: { url: 'https://a.hermes.cloud', authMode: 'oauth', org: 'nous' }
  })

  const cloud = registry.connections.find(c => c.kind === 'cloud')

  assert.ok(cloud)
  assert.equal(registry.primary, cloud.id)
  assert.equal(cloud.org, 'nous')
})

test('migrate: per-profile overrides become extra sources, deduped by URL', () => {
  const registry = migrateV1ToRegistry({
    mode: 'remote',
    remote: { url: 'http://homelab.lan:9119', authMode: 'token', token: { enc: 'x' } },
    profiles: {
      research: { mode: 'remote', url: 'http://homelab.lan:9119', authMode: 'token', token: { enc: 'x' } },
      coder: { mode: 'remote', url: 'http://other.lan:9119', authMode: 'token', token: { enc: 'y' } },
      sparky: { mode: 'ssh', host: 'spark1', user: 'tek' },
      plain: { mode: 'local', savedSsh: { mode: 'ssh', host: 'spark1', user: 'tek' } }
    }
  })

  // homelab (global+research deduped), other.lan, spark ssh (override+savedSsh deduped), local
  assert.equal(registry.connections.length, 4)
  assert.equal(registry.connections.filter(c => c.kind === 'remote').length, 2)
  assert.equal(registry.connections.filter(c => c.kind === 'ssh').length, 1)
})

test('migrate: v1 global ssh becomes the primary', () => {
  const registry = migrateV1ToRegistry({
    mode: 'ssh',
    remote: { mode: 'ssh', host: 'spark1', user: 'tek', port: 2222 }
  })

  const ssh = registry.connections.find(c => c.kind === 'ssh')

  assert.ok(ssh)
  assert.equal(registry.primary, ssh.id)
  assert.equal(ssh.label, 'spark1')
})

test('migrate: duplicate host labels are suffixed, not dropped', () => {
  const registry = migrateV1ToRegistry({
    mode: 'remote',
    remote: { url: 'http://box.lan:9119', authMode: 'token', token: {} },
    profiles: {
      a: { mode: 'ssh', host: 'box.lan' }
    }
  })

  const labels = registry.connections.map(c => labelKey(c.label))

  assert.equal(new Set(labels).size, labels.length)
  assert.equal(registry.connections.length, 3)
})

// --- registry operations ---

test('removeConnection: local refuses, primary retargets to local', () => {
  let registry = emptyRegistry()
  const entry = normalizeConnectionInput({ kind: 'remote', label: 'Homelab', url: 'http://10.0.0.5:9119' }, registry)
  registry = upsertConnection(registry, entry)
  registry = setPrimaryConnection(registry, entry.id)

  assert.throws(() => removeConnection(registry, LOCAL_CONNECTION_ID), /cannot be removed/)

  const after = removeConnection(registry, entry.id)

  assert.equal(after.primary, LOCAL_CONNECTION_ID)
  assert.equal(after.connections.length, 1)
  // Removing an unknown id is a no-op, not an error.
  assert.equal(removeConnection(after, 'ghost'), after)
})

test('setPrimaryConnection validates the target id', () => {
  const registry = emptyRegistry()

  assert.throws(() => setPrimaryConnection(registry, 'ghost'), /No connection/)
  assert.equal(setPrimaryConnection(registry, LOCAL_CONNECTION_ID).primary, LOCAL_CONNECTION_ID)
})

test('upsertConnection replaces by id and appends new ids', () => {
  let registry = emptyRegistry()
  const a = normalizeConnectionInput({ kind: 'remote', label: 'A', url: 'http://a:1' }, registry)
  registry = upsertConnection(registry, a)
  registry = upsertConnection(registry, { ...a, url: 'http://a:2' })

  assert.equal(registry.connections.filter(c => c.id === a.id).length, 1)
  assert.equal(registry.connections.find(c => c.id === a.id)?.url, 'http://a:2')
})

// --- connectionDialFieldsChanged (edit → recycle decision) ---

test('connectionDialFieldsChanged: label-only edits do not recycle', () => {
  const before = {
    id: 'homelab',
    kind: 'remote',
    label: 'Homelab',
    url: 'http://10.0.0.5:9119',
    authMode: 'token',
    token: { encoding: 'safeStorage', value: 'abc' }
  } as const

  assert.equal(connectionDialFieldsChanged(before, { ...before, label: 'Home lab (renamed)' }), false)
  // Identity edit is also a no-op.
  assert.equal(connectionDialFieldsChanged(before, { ...before }), false)
})

test('connectionDialFieldsChanged: url / auth / token changes recycle', () => {
  const before = {
    id: 'homelab',
    kind: 'remote',
    label: 'Homelab',
    url: 'http://10.0.0.5:9119',
    authMode: 'token',
    token: { encoding: 'safeStorage', value: 'abc' }
  } as const

  assert.equal(connectionDialFieldsChanged(before, { ...before, url: 'http://10.0.0.9:9119' }), true)
  assert.equal(connectionDialFieldsChanged(before, { ...before, authMode: 'oauth', token: undefined }), true)
  assert.equal(
    connectionDialFieldsChanged(before, { ...before, token: { encoding: 'safeStorage', value: 'NEW' } }),
    true
  )
})

test('connectionDialFieldsChanged: ssh routing fields recycle, kind change recycles', () => {
  const before = { id: 'box', kind: 'ssh', label: 'Box', host: 'box.lan', user: 'me', port: 22 } as const

  assert.equal(connectionDialFieldsChanged(before, { ...before, label: 'Box 2' }), false)
  assert.equal(connectionDialFieldsChanged(before, { ...before, host: 'other.lan' }), true)
  assert.equal(connectionDialFieldsChanged(before, { ...before, port: 2222 }), true)
  assert.equal(connectionDialFieldsChanged(before, { ...before, remoteProfile: 'work' }), true)
  assert.equal(
    connectionDialFieldsChanged(before, { id: 'box', kind: 'remote', label: 'Box', url: 'http://x:1' }),
    true
  )
})

// --- remote gateway headers (Cloudflare Access etc., #74466 / PR #74468) ---

test('normalizeConnectionInput keeps filtered headers on remote/cloud, drops them elsewhere', () => {
  const registry = emptyRegistry()

  const remote = normalizeConnectionInput(
    {
      kind: 'remote',
      label: 'CF box',
      url: 'https://hermes.example.com',
      authMode: 'token',
      token: { enc: 'x' },
      headers: {
        'CF-Access-Client-Id': { encoding: 'safeStorage', value: 'id' },
        Authorization: { encoding: 'plain', value: 'blocked' }
      }
    },
    registry
  )

  assert.deepEqual(remote.headers, {
    'CF-Access-Client-Id': { encoding: 'safeStorage', value: 'id' }
  })

  const ssh = normalizeConnectionInput(
    {
      kind: 'ssh',
      label: 'Box',
      host: 'box.lan',
      headers: { 'CF-Access-Client-Id': { encoding: 'plain', value: 'id' } }
    } as any,
    registry
  )

  assert.equal((ssh as any).headers, undefined)
})

test('mergeConnectionInput inherits stored headers when the editor payload omits them', () => {
  const stored = {
    id: 'cf',
    kind: 'remote' as const,
    label: 'CF box',
    url: 'https://hermes.example.com',
    authMode: 'token' as const,
    headers: { 'CF-Access-Client-Id': { encoding: 'safeStorage', value: 'id' } }
  }

  const renamed = mergeConnectionInput({ id: 'cf', kind: 'remote', label: 'Renamed' }, stored)

  assert.deepEqual(renamed.headers, stored.headers)

  // An explicit headers payload (even empty) is authoritative — clearing works.
  const cleared = mergeConnectionInput({ id: 'cf', kind: 'remote', label: 'CF box', headers: {} }, stored)

  assert.deepEqual(cleared.headers, {})
})

test('connectionDialFieldsChanged: a header change recycles live backends', () => {
  const before = {
    id: 'cf',
    kind: 'remote',
    label: 'CF box',
    url: 'https://hermes.example.com',
    authMode: 'token',
    token: { enc: 'x' },
    headers: { 'CF-Access-Client-Id': { encoding: 'safeStorage', value: 'id' } }
  } as const

  assert.equal(connectionDialFieldsChanged(before, { ...before }), false)
  assert.equal(
    connectionDialFieldsChanged(before, {
      ...before,
      headers: { 'CF-Access-Client-Id': { encoding: 'safeStorage', value: 'OTHER' } }
    }),
    true
  )
  assert.equal(connectionDialFieldsChanged(before, { ...before, headers: undefined }), true)
})

test('normalizeRegistry preserves stored headers on remote entries (v2 additive field)', () => {
  const registry = normalizeRegistry({
    version: REGISTRY_VERSION,
    primary: 'cf',
    connections: [
      { id: 'local', kind: 'local', label: 'This device' },
      {
        id: 'cf',
        kind: 'remote',
        label: 'CF box',
        url: 'https://hermes.example.com',
        authMode: 'token',
        token: { enc: 'x' },
        headers: {
          'CF-Access-Client-Id': { encoding: 'safeStorage', value: 'id' },
          Cookie: { encoding: 'plain', value: 'blocked' }
        }
      }
    ]
  })

  const remote = registry.connections.find(c => c.id === 'cf')

  assert.ok(remote)
  assert.deepEqual(remote.headers, {
    'CF-Access-Client-Id': { encoding: 'safeStorage', value: 'id' }
  })
})

test('migrateV1ToRegistry carries v1 remote headers into the registry entry', () => {
  const registry = migrateV1ToRegistry({
    mode: 'remote',
    remote: {
      url: 'https://hermes.example.com',
      authMode: 'token',
      token: { enc: 'x' },
      headers: { 'CF-Access-Client-Id': { encoding: 'safeStorage', value: 'id' } }
    }
  })

  const remote = registry.connections.find(c => c.kind === 'remote')

  assert.ok(remote)
  assert.deepEqual(remote.headers, {
    'CF-Access-Client-Id': { encoding: 'safeStorage', value: 'id' }
  })
})
