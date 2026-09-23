/**
 * Tests for electron/connection-registry.ts connection resolution, labels,
 * backend scopes, local routing, SSH inventory, and the agent roster.
 */

import assert from 'node:assert/strict'

import { test } from 'vitest'

import type { ConnectionRegistry } from './connection-registry'
import {
  agentHandle,
  backendScopeKey,
  backendScopePrefix,
  buildAgentRoster,
  connectionIdForLabel,
  labelKey,
  labelSlug,
  LOCAL_CONNECTION_ID,
  mergeConnectionInput,
  migrateV1ToRegistry,
  normalizeConnectionInput,
  normalizeRegistry,
  parseRemoteProfileListing,
  reconcileAppliedGlobalConnection,
  REGISTRY_VERSION,
  registrySourceOwnsPrimaryBackend,
  rememberSshEnumeration,
  resolvedConnectionId,
  resolveRegistryLocalRoute,
  reuseMatchingPrimarySshBackend,
  shouldDeferLocalEnumeration,
  shouldRetrySshInventory,
  uniqueLabel,
  upsertConnection
} from './connection-registry'

function emptyRegistry(): ConnectionRegistry {
  return normalizeRegistry(null)
}

test('Cloud apply upgrades only host labels without changing connection identity', () => {
  const url = 'https://agent.example.com'
  const name = 'Research cloud'

  const first = reconcileAppliedGlobalConnection(emptyRegistry(), {
    mode: 'cloud',
    remote: { url, authMode: 'oauth' }
  })

  const named = reconcileAppliedGlobalConnection(first, {
    mode: 'cloud',
    remote: { url, authMode: 'oauth', name }
  })

  assert.equal(named.primary, first.primary)
  assert.equal(named.connections.find(c => c.id === named.primary)?.label, name)
  const restored = normalizeRegistry(JSON.parse(JSON.stringify(named)))
  assert.equal(restored.connections.find(c => c.id === named.primary)?.name, name)

  const custom = upsertConnection(restored, {
    ...restored.connections.find(c => c.id === named.primary)!,
    label: 'My device'
  })

  const reapplied = reconcileAppliedGlobalConnection(custom, {
    mode: 'cloud',
    remote: { url, authMode: 'oauth', name: 'New portal name' }
  })

  assert.equal(reapplied.primary, first.primary)
  assert.equal(reapplied.connections.find(c => c.id === first.primary)?.label, 'My device')

  const other = reconcileAppliedGlobalConnection(reapplied, {
    mode: 'cloud',
    remote: { url: 'https://other.example.com', authMode: 'oauth' }
  })

  assert.notEqual(other.primary, first.primary)
  assert.equal(other.connections.find(c => c.id === other.primary)?.name, undefined)
})

test('Cloud name survives partial edits but never inherits across gateway URLs', () => {
  const registry = reconcileAppliedGlobalConnection(emptyRegistry(), {
    mode: 'cloud',
    remote: { url: 'https://agent.example.com', authMode: 'oauth', name: 'Research cloud' }
  })

  const existing = registry.connections.find(c => c.id === registry.primary)!

  const renamed = normalizeConnectionInput(
    mergeConnectionInput({ id: existing.id, kind: 'cloud', label: 'Mine' }, existing),
    registry
  )

  assert.equal(renamed.name, 'Research cloud')

  const retargeted = normalizeConnectionInput(
    mergeConnectionInput({ id: existing.id, kind: 'cloud', label: 'Mine', url: 'https://other.example.com' }, existing),
    registry
  )

  assert.equal(retargeted.name, undefined)
})

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

test('matching primary/default SSH route reuses the existing descriptor once', async () => {
  const registry = migrateV1ToRegistry({
    mode: 'ssh',
    remote: { mode: 'ssh', host: 'build-host', user: 'alice' },
    profiles: {}
  })

  const source = registry.connections.find(connection => connection.id === registry.primary)

  const descriptor = {
    mode: 'remote' as const,
    remoteKind: 'ssh' as const,
    ssh: {
      effectiveConfigFingerprint: 'same-effective-config',
      host: 'build-host',
      keyPath: '~/.ssh/id_ed25519',
      remoteProfile: 'default',
      user: 'alice'
    }
  }

  let ensureCalls = 0
  let fingerprintCalls = 0

  assert.equal(source?.kind, 'ssh')
  assert.equal(
    await reuseMatchingPrimarySshBackend({
      connectionId: registry.primary,
      effectiveFingerprint: async () => {
        fingerprintCalls += 1

        return 'same-effective-config'
      },
      ensurePrimary: async () => {
        ensureCalls += 1

        return descriptor
      },
      profile: 'default',
      registry,
      source: source!
    }),
    descriptor
  )
  assert.equal(ensureCalls, 1)
  assert.equal(fingerprintCalls, 1)
})

test('non-default or non-primary SSH routes do not resolve the primary backend', async () => {
  const registry = migrateV1ToRegistry({
    mode: 'ssh',
    remote: { mode: 'ssh', host: 'build-host', user: 'alice' },
    profiles: {}
  })

  const source = registry.connections.find(connection => connection.id === registry.primary)!
  let ensureCalls = 0

  const opts = {
    effectiveFingerprint: async () => 'same',
    ensurePrimary: async () => {
      ensureCalls += 1

      return { mode: 'remote' as const, remoteKind: 'ssh' as const }
    },
    registry,
    source
  }

  assert.equal(
    await reuseMatchingPrimarySshBackend({ ...opts, connectionId: registry.primary, profile: 'researcher' }),
    null
  )
  assert.equal(
    await reuseMatchingPrimarySshBackend({ ...opts, connectionId: LOCAL_CONNECTION_ID, profile: 'default' }),
    null
  )
  assert.equal(ensureCalls, 0)
})

test('primary SSH reuse rejects a descriptor with different effective dialing config', async () => {
  const registry = migrateV1ToRegistry({
    mode: 'ssh',
    remote: { mode: 'ssh', host: 'build-host', user: 'alice' },
    profiles: {}
  })

  const source = registry.connections.find(connection => connection.id === registry.primary)!

  assert.equal(
    await reuseMatchingPrimarySshBackend({
      connectionId: registry.primary,
      effectiveFingerprint: async () => 'registry-config',
      ensurePrimary: async () => ({
        mode: 'remote',
        remoteKind: 'ssh',
        ssh: {
          effectiveConfigFingerprint: 'active-config',
          host: 'other-host',
          remoteProfile: ''
        }
      }),
      profile: 'default',
      registry,
      source
    }),
    null
  )
})

test('primary SSH reuse rejects a descriptor with a different remote Hermes path', async () => {
  const registry = migrateV1ToRegistry({
    mode: 'ssh',
    remote: { mode: 'ssh', host: 'build-host', remoteHermesPath: '/srv/hermes', user: 'alice' },
    profiles: {}
  })

  const source = registry.connections.find(connection => connection.id === registry.primary)!

  assert.equal(
    await reuseMatchingPrimarySshBackend({
      connectionId: registry.primary,
      effectiveFingerprint: async () => 'same-effective-config',
      ensurePrimary: async () => ({
        mode: 'remote',
        remoteKind: 'ssh',
        ssh: {
          effectiveConfigFingerprint: 'same-effective-config',
          host: 'build-host',
          remoteHermesPath: '/opt/hermes',
          remoteProfile: '',
          user: 'alice'
        }
      }),
      profile: 'default',
      registry,
      source
    }),
    null
  )
})

test('registry primary reuses a matching primary backend descriptor', () => {
  const registry = normalizeRegistry({
    version: REGISTRY_VERSION,
    primary: 'hermes-vps',
    launchMode: 'primary',
    lastUsed: 'hermes-vps',
    connections: [
      { id: LOCAL_CONNECTION_ID, kind: 'local', label: 'This device' },
      { id: 'hermes-vps', kind: 'ssh', label: 'Hermes VPS', host: 'hermes-vps' }
    ]
  })

  const descriptor = {
    connectionId: 'hermes-vps',
    mode: 'remote' as const,
    remoteKind: 'ssh' as const,
    ssh: { host: 'hermes-vps' }
  }

  assert.equal(registrySourceOwnsPrimaryBackend(registry, 'hermes-vps', descriptor), true)
  assert.equal(registrySourceOwnsPrimaryBackend(registry, LOCAL_CONNECTION_ID, descriptor), false)
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
      remoteHost: 'ROOT@WORK-HOST',
      remoteKind: 'ssh'
    }),
    work?.id
  )

  const ambiguousLocal: ConnectionRegistry = {
    ...registry,
    connections: [...registry.connections, { id: 'local-copy', kind: 'local', label: 'Local copy' }]
  }

  assert.equal(resolvedConnectionId(ambiguousLocal, { mode: 'local' }), null)
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

test('resolvedConnectionId accepts only a current exact descriptor id and never falls back', () => {
  const registry: ConnectionRegistry = {
    version: REGISTRY_VERSION,
    primary: 'remote-a',
    launchMode: 'primary',
    lastUsed: 'remote-a',
    connections: [
      { id: LOCAL_CONNECTION_ID, kind: 'local', label: 'This device' },
      {
        id: 'remote-a',
        kind: 'remote',
        label: 'Remote A',
        url: 'https://shared.example',
        authMode: 'token',
        token: { encoding: 'safeStorage', value: 'token-a' },
        headers: { 'CF-Access-Client-Id': { encoding: 'safeStorage', value: 'header-a' } }
      },
      {
        id: 'remote-b',
        kind: 'remote',
        label: 'Remote B',
        url: 'https://shared.example',
        authMode: 'oauth',
        headers: { 'CF-Access-Client-Id': { encoding: 'safeStorage', value: 'header-b' } }
      }
    ]
  }

  assert.equal(
    resolvedConnectionId(registry, {
      authMode: 'token',
      baseUrl: 'https://shared.example',
      connectionId: 'remote-b',
      headers: { 'CF-Access-Client-Id': { encoding: 'safeStorage', value: 'header-a' } },
      mode: 'remote',
      remoteKind: 'url',
      token: { encoding: 'safeStorage', value: 'token-a' }
    }),
    'remote-b'
  )

  const inferableRemoteA = {
    authMode: 'token',
    baseUrl: 'https://shared.example',
    headers: { 'CF-Access-Client-Id': { encoding: 'safeStorage', value: 'header-a' } },
    mode: 'remote' as const,
    remoteKind: 'url' as const,
    token: { encoding: 'safeStorage', value: 'token-a' }
  }

  // Only true absence enters compatibility inference. Every explicitly
  // present invalid value remains unresolved even though the remaining
  // envelope uniquely identifies remote-a.
  assert.equal(resolvedConnectionId(registry, inferableRemoteA), 'remote-a')

  for (const connectionId of ['', '   ', null, undefined, 42, {}, 'unknown-source', 'retired-source']) {
    assert.equal(resolvedConnectionId(registry, { ...inferableRemoteA, connectionId }), null)
  }

  // Registry order is never authority for either an exact current id or a
  // rejected explicit claim.
  const reordered = { ...registry, connections: [...registry.connections].reverse() }

  assert.equal(
    resolvedConnectionId(reordered, {
      ...inferableRemoteA,
      connectionId: 'remote-b'
    }),
    'remote-b'
  )
  assert.equal(resolvedConnectionId(reordered, { ...inferableRemoteA, connectionId: 'retired-source' }), null)
})

test('resolvedConnectionId reuses the exact URL envelope and rejects weak or duplicate matches', () => {
  const sharedUrl = 'https://shared.example/gateway'

  const registry: ConnectionRegistry = {
    version: REGISTRY_VERSION,
    primary: 'remote-token',
    launchMode: 'primary',
    lastUsed: 'remote-token',
    connections: [
      { id: LOCAL_CONNECTION_ID, kind: 'local', label: 'This device' },
      {
        id: 'remote-token',
        kind: 'remote',
        label: 'Token remote',
        url: sharedUrl,
        authMode: 'token',
        token: { encoding: 'safeStorage', value: 'token-a' },
        headers: { 'CF-Access-Client-Id': { encoding: 'safeStorage', value: 'header-a' } }
      },
      {
        id: 'remote-oauth',
        kind: 'remote',
        label: 'OAuth remote',
        url: `${sharedUrl}/`,
        authMode: 'oauth',
        headers: { 'CF-Access-Client-Id': { encoding: 'safeStorage', value: 'header-b' } }
      },
      {
        id: 'cloud-nous',
        kind: 'cloud',
        label: 'Nous cloud',
        url: sharedUrl,
        authMode: 'oauth',
        headers: { 'CF-Access-Client-Id': { encoding: 'safeStorage', value: 'header-cloud' } },
        org: 'nous'
      },
      {
        id: 'cloud-labs',
        kind: 'cloud',
        label: 'Labs cloud',
        url: sharedUrl,
        authMode: 'oauth',
        headers: { 'CF-Access-Client-Id': { encoding: 'safeStorage', value: 'header-cloud' } },
        org: 'labs'
      }
    ]
  }

  assert.equal(
    resolvedConnectionId(registry, {
      authMode: 'token',
      baseUrl: sharedUrl,
      headers: { 'cf-access-client-id': { encoding: 'safeStorage', value: 'header-a' } },
      mode: 'remote',
      remoteKind: 'url',
      token: { encoding: 'safeStorage', value: 'token-a' }
    }),
    'remote-token'
  )
  assert.equal(
    resolvedConnectionId(registry, {
      authMode: 'oauth',
      baseUrl: sharedUrl,
      headers: { 'CF-ACCESS-CLIENT-ID': { encoding: 'safeStorage', value: 'header-b' } },
      mode: 'remote',
      remoteKind: 'url'
    }),
    'remote-oauth'
  )
  assert.equal(
    resolvedConnectionId(registry, {
      authMode: 'oauth',
      baseUrl: sharedUrl,
      headers: { 'CF-Access-Client-Id': { encoding: 'safeStorage', value: 'header-cloud' } },
      mode: 'remote',
      org: 'nous',
      remoteKind: 'cloud'
    }),
    'cloud-nous'
  )
  assert.equal(
    resolvedConnectionId(registry, {
      authMode: 'oauth',
      baseUrl: sharedUrl,
      headers: { 'CF-Access-Client-Id': { encoding: 'safeStorage', value: 'header-cloud' } },
      mode: 'remote',
      org: 'labs',
      remoteKind: 'cloud'
    }),
    'cloud-labs'
  )

  // Post-dial URL-only shapes do not contain enough proof to choose a source.
  assert.equal(resolvedConnectionId(registry, { baseUrl: sharedUrl, mode: 'remote', remoteKind: 'url' }), null)
  assert.equal(resolvedConnectionId(registry, { baseUrl: sharedUrl, mode: 'remote', remoteKind: 'cloud' }), null)

  // Even a complete envelope fails closed when two registrations are exact twins.
  const duplicate: ConnectionRegistry = {
    ...registry,
    connections: [
      ...registry.connections,
      { ...registry.connections.find(connection => connection.id === 'remote-token')!, id: 'remote-token-copy' }
    ]
  }

  assert.equal(
    resolvedConnectionId(duplicate, {
      authMode: 'token',
      baseUrl: sharedUrl,
      headers: { 'CF-Access-Client-Id': { encoding: 'safeStorage', value: 'header-a' } },
      mode: 'remote',
      remoteKind: 'url',
      token: { encoding: 'safeStorage', value: 'token-a' }
    }),
    null
  )
  assert.equal(
    resolvedConnectionId(
      { ...duplicate, connections: [...duplicate.connections].reverse() },
      {
        authMode: 'token',
        baseUrl: sharedUrl,
        headers: { 'CF-Access-Client-Id': { encoding: 'safeStorage', value: 'header-a' } },
        mode: 'remote',
        remoteKind: 'url',
        token: { encoding: 'safeStorage', value: 'token-a' }
      }
    ),
    null
  )
})

test('resolvedConnectionId keeps same-host SSH routes distinct by port, key, path, and profile', () => {
  const base = {
    host: 'work-host',
    keyPath: '/keys/a',
    kind: 'ssh' as const,
    remoteHermesPath: '/srv/hermes',
    remoteProfile: 'alpha',
    user: 'root'
  }

  const registry: ConnectionRegistry = {
    version: REGISTRY_VERSION,
    primary: 'ssh-base',
    launchMode: 'primary',
    lastUsed: 'ssh-base',
    connections: [
      { id: LOCAL_CONNECTION_ID, kind: 'local', label: 'This device' },
      { ...base, id: 'ssh-base', label: 'SSH base' },
      { ...base, id: 'ssh-port', label: 'SSH port', port: 2222 },
      { ...base, id: 'ssh-key', keyPath: '/keys/b', label: 'SSH key' },
      { ...base, id: 'ssh-path', label: 'SSH path', remoteHermesPath: '/opt/hermes' },
      { ...base, id: 'ssh-profile', label: 'SSH profile', remoteProfile: 'beta' }
    ]
  }

  const resolve = (ssh: NonNullable<Parameters<typeof resolvedConnectionId>[1]['ssh']>) =>
    resolvedConnectionId(registry, { mode: 'remote', remoteKind: 'ssh', ssh })

  assert.equal(resolve(base), 'ssh-base')
  assert.equal(resolve({ ...base, port: 2222 }), 'ssh-port')
  assert.equal(resolve({ ...base, keyPath: '/keys/b' }), 'ssh-key')
  assert.equal(resolve({ ...base, remoteHermesPath: '/opt/hermes' }), 'ssh-path')
  assert.equal(resolve({ ...base, remoteProfile: 'beta' }), 'ssh-profile')
  assert.equal(
    resolvedConnectionId(registry, {
      connectionId: 'ssh-port',
      mode: 'remote',
      remoteKind: 'ssh',
      ssh: { ...base, port: 9999 }
    }),
    'ssh-port'
  )

  // user@host is a transport hint, not a registry identity, when variants coexist.
  assert.equal(
    resolvedConnectionId(registry, {
      mode: 'remote',
      remoteHost: 'ROOT@WORK-HOST',
      remoteKind: 'ssh'
    }),
    null
  )
  assert.equal(
    resolvedConnectionId(registry, {
      mode: 'remote',
      remoteHost: 'ROOT@WORK-HOST',
      remoteKind: 'ssh',
      ssh: undefined
    }),
    null
  )

  const duplicate: ConnectionRegistry = {
    ...registry,
    connections: [...registry.connections, { ...base, id: 'ssh-base-copy', label: 'SSH base copy' }]
  }

  assert.equal(resolvedConnectionId(duplicate, { mode: 'remote', remoteKind: 'ssh', ssh: base }), null)
  assert.equal(
    resolvedConnectionId(
      { ...duplicate, connections: [...duplicate.connections].reverse() },
      { mode: 'remote', remoteKind: 'ssh', ssh: base }
    ),
    null
  )
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

test('registry local route: a per-profile remote override delegates to the override (#90477)', () => {
  // The per-profile SSH/remote override is the authoritative route for that
  // profile. Forcing local here made the roster list the profile via its
  // override but open the thread in a local child — which fails when the
  // profile exists only on the remote. The override must win.
  const route = resolveRegistryLocalRoute('research', { profileRemoteOverride: true })

  assert.deepEqual(route, { delegate: true, poolKey: 'research' })
})

test('registry local route: per-profile override wins when global remote is also active', () => {
  const route = resolveRegistryLocalRoute('research', {
    globalRemote: true,
    profileRemoteOverride: true
  })

  assert.deepEqual(route, { delegate: true, poolKey: 'research' })
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

test('roster: source profile metadata follows the connection-qualified row', () => {
  const local = { id: 'local', kind: 'local' as const, label: 'This device' }
  const vps = { id: 'vps', kind: 'remote' as const, label: 'VPS', url: 'http://vps:8642' }

  const vpsMeta = {
    display_name: 'Emma',
    ui_meta: { 'hermes-bots': { title: 'Emma', shape: 'blobatar::sun', color: '#8b5cf6' } },
    has_avatar: true
  }

  const roster = buildAgentRoster([
    { connection: local, profiles: ['default'] },
    { connection: vps, profiles: ['default'], profileMetadata: { default: vpsMeta } }
  ])

  assert.deepEqual(roster.find(agent => agent.connectionId === 'vps')?.profileMetadata, vpsMeta)
  assert.equal(roster.find(agent => agent.connectionId === 'local')?.profileMetadata, undefined)
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

test('rememberSshEnumeration: a bounced remote source keeps its last-known roster (4-bots-show-as-2)', () => {
  // A VPS restart makes the remote source unreachable for a few polls. The
  // last successful enumeration must keep painting so the roster does not
  // silently drop that source's bots mid-outage.
  assert.deepEqual(
    rememberSshEnumeration({ profiles: null, error: 'unreachable' }, ['default', 'ceo', 'accounter'], 'remote'),
    { profiles: ['default', 'ceo', 'accounter'], error: 'unreachable' }
  )
  // Never-seen remote source: no seed — an unreachable URL is not evidence a
  // backend exists there.
  assert.deepEqual(rememberSshEnumeration({ profiles: null, error: 'unreachable' }, null, 'remote'), {
    profiles: null,
    error: 'unreachable'
  })
  // Local enumeration failures never reuse a cache (the local runtime answers
  // authoritatively or not at all).
  assert.deepEqual(rememberSshEnumeration({ profiles: null, error: 'boom' }, ['default'], 'local'), {
    profiles: null,
    error: 'boom'
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
