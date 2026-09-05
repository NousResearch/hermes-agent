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
  cloudLabelIsDerived,
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
  reconcileAppliedGlobalConnection,
  reconcileCloudInstanceNames,
  reconcileRegistryDrift,
  REGISTRY_VERSION,
  registrySourceOwnsPrimaryBackend,
  rememberSshEnumeration,
  removeConnection,
  resolvedConnectionId,
  resolveRegistryLocalRoute,
  reuseMatchingPrimarySshBackend,
  setConnectionLaunchMode,
  setLastUsedConnection,
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

test('registry SSH fingerprint failures name the connection and ssh -G step', async () => {
  const registry = migrateV1ToRegistry({
    mode: 'ssh',
    remote: { mode: 'ssh', host: 'build-host', user: 'alice' },
    profiles: {}
  })

  const source = registry.connections.find(connection => connection.id === registry.primary)!

  const cause = new Error('spawn ssh ENOENT')

  source.label = 'Build box'

  await assert.rejects(
    reuseMatchingPrimarySshBackend({
      connectionId: registry.primary,
      effectiveFingerprint: async () => {
        throw cause
      },
      ensurePrimary: async () => ({ mode: 'remote', remoteKind: 'ssh' }),
      profile: 'default',
      registry,
      source
    }),
    error => {
      assert.equal(
        (error as Error).message,
        `Could not resolve effective SSH config for connection "Build box" (${source.id}) via ssh -G: spawn ssh ENOENT`
      )
      assert.equal((error as Error).cause, cause)

      return true
    }
  )
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
    assert.equal(registry.launchMode, 'primary')
    assert.equal(registry.lastUsed, LOCAL_CONNECTION_ID)
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
    launchMode: 'last-used',
    lastUsed: 'homelab',
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
  assert.equal(registry.launchMode, 'last-used')
  assert.equal(registry.lastUsed, 'homelab')
  assert.equal(registry.connections.length, 4)
  assert.deepEqual(
    registry.connections.map(c => c.id),
    ['local', 'homelab', 'cloud-1', 'spark']
  )
  assert.deepEqual(registry.connections[1].token, { v: 1 })
  assert.equal(registry.connections[3].port, 2222)
})

test('normalizeRegistry falls back to Primary when the last-used source is missing', () => {
  const registry = normalizeRegistry({
    version: 2,
    primary: 'homelab',
    launchMode: 'last-used',
    lastUsed: 'retired-host',
    connections: [
      { id: 'local', kind: 'local', label: 'This device' },
      { id: 'homelab', kind: 'remote', label: 'Homelab', url: 'http://10.0.0.5:9119' }
    ]
  })

  assert.equal(registry.launchMode, 'last-used')
  assert.equal(registry.lastUsed, 'homelab')
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

test('removeConnection: local refuses, primary and last-used retarget safely', () => {
  let registry = emptyRegistry()
  const entry = normalizeConnectionInput({ kind: 'remote', label: 'Homelab', url: 'http://10.0.0.5:9119' }, registry)
  registry = upsertConnection(registry, entry)
  registry = setPrimaryConnection(registry, entry.id)
  registry = setLastUsedConnection(registry, entry.id)

  assert.throws(() => removeConnection(registry, LOCAL_CONNECTION_ID), /cannot be removed/)

  const after = removeConnection(registry, entry.id)

  assert.equal(after.primary, LOCAL_CONNECTION_ID)
  assert.equal(after.lastUsed, LOCAL_CONNECTION_ID)
  assert.equal(after.connections.length, 1)
  // Removing an unknown id is a no-op, not an error.
  assert.equal(removeConnection(after, 'ghost'), after)
})

test('setPrimaryConnection validates the target id', () => {
  const registry = emptyRegistry()

  assert.throws(() => setPrimaryConnection(registry, 'ghost'), /No connection/)
  assert.equal(setPrimaryConnection(registry, LOCAL_CONNECTION_ID).primary, LOCAL_CONNECTION_ID)
})

test('last-used source and launch mode validate their persisted values', () => {
  let registry = emptyRegistry()
  const entry = normalizeConnectionInput({ kind: 'remote', label: 'Homelab', url: 'http://10.0.0.5:9119' }, registry)
  registry = upsertConnection(registry, entry)

  assert.throws(() => setLastUsedConnection(registry, 'ghost'), /No connection/)
  assert.equal(setLastUsedConnection(registry, entry.id).lastUsed, entry.id)
  assert.equal(setConnectionLaunchMode(registry, 'last-used').launchMode, 'last-used')
  assert.throws(() => setConnectionLaunchMode(registry, 'sometimes'), /Unknown connection launch mode/)
})

test('upsertConnection replaces by id and appends new ids', () => {
  let registry = emptyRegistry()
  const a = normalizeConnectionInput({ kind: 'remote', label: 'A', url: 'http://a:1' }, registry)
  registry = upsertConnection(registry, a)
  registry = upsertConnection(registry, { ...a, url: 'http://a:2' })

  assert.equal(registry.connections.filter(c => c.id === a.id).length, 1)
  assert.equal(registry.connections.find(c => c.id === a.id)?.url, 'http://a:2')
})

test('Apply remote inserts into an existing local-only registry and becomes primary/current', () => {
  const registry = reconcileAppliedGlobalConnection(emptyRegistry(), {
    mode: 'remote',
    remote: { url: 'https://gateway.example.com/', authMode: 'oauth' }
  })

  const remote = registry.connections.find(connection => connection.kind === 'remote')

  assert.ok(remote)
  assert.equal(registry.primary, remote.id)
  assert.equal(registry.lastUsed, remote.id)
  assert.equal(
    resolvedConnectionId(registry, {
      authMode: 'oauth',
      baseUrl: 'https://gateway.example.com',
      headers: {},
      mode: 'remote',
      remoteKind: 'url'
    }),
    remote.id
  )
})

test('Apply remote preserves an existing URL identity and label without duplicates', () => {
  let registry = emptyRegistry()

  registry = upsertConnection(registry, {
    id: 'hermes-alex',
    kind: 'remote',
    label: 'Existing gateway',
    url: 'https://gateway.example.com',
    authMode: 'token',
    token: { old: true }
  })

  const applied = reconcileAppliedGlobalConnection(registry, {
    mode: 'remote',
    remote: { url: 'https://GATEWAY.example.com/', authMode: 'oauth' }
  })

  const matches = applied.connections.filter(connection => connection.url === 'https://gateway.example.com')

  assert.equal(matches.length, 1)
  assert.equal(matches[0].id, 'hermes-alex')
  assert.equal(matches[0].label, 'Existing gateway')
  assert.equal(matches[0].authMode, 'oauth')
  assert.equal(applied.primary, 'hermes-alex')
  assert.equal(applied.lastUsed, 'hermes-alex')
})

// --- Hermes Cloud instance names ---

const CLOUD_URL = 'https://agent-7f3a.hermes.nousresearch.com'

function appliedCloud(registry: ConnectionRegistry, name?: string): ConnectionRegistry {
  return reconcileAppliedGlobalConnection(registry, {
    mode: 'cloud',
    remote: { url: CLOUD_URL, authMode: 'oauth', org: 'nous', ...(name === undefined ? {} : { name }) }
  })
}

test('Apply cloud labels the connection with the instance name, not the dashboard host', () => {
  const registry = appliedCloud(emptyRegistry(), 'Research Bot')
  const cloud = registry.connections.find(connection => connection.kind === 'cloud')

  assert.ok(cloud)
  assert.equal(cloud.label, 'Research Bot')
  assert.equal(cloud.cloudName, 'Research Bot')
  assert.equal(cloud.url, CLOUD_URL)
})

test('Apply cloud still falls back to the host when the instance has no name', () => {
  const cloud = appliedCloud(emptyRegistry()).connections.find(connection => connection.kind === 'cloud')

  assert.ok(cloud)
  assert.equal(cloud.label, 'agent-7f3a.hermes.nousresearch.com')
  assert.equal(cloud.cloudName, undefined)
})

test('Apply cloud replaces a host-derived label once the instance name is known', () => {
  const registry = appliedCloud(emptyRegistry())
  const renamed = appliedCloud(registry, 'Research Bot')
  const cloud = renamed.connections.find(connection => connection.kind === 'cloud')

  assert.ok(cloud)
  assert.equal(cloud.label, 'Research Bot')
  // Same entry, not a second registration for the same gateway.
  assert.equal(renamed.connections.filter(connection => connection.kind === 'cloud').length, 1)
})

test('Apply cloud never overwrites a device name the user typed here', () => {
  let registry = appliedCloud(emptyRegistry(), 'Research Bot')
  const cloudId = registry.connections.find(connection => connection.kind === 'cloud')!.id

  registry = upsertConnection(registry, {
    ...registry.connections.find(connection => connection.id === cloudId)!,
    label: 'Work bot'
  })

  const cloud = appliedCloud(registry, 'Renamed In Portal').connections.find(connection => connection.id === cloudId)

  assert.ok(cloud)
  assert.equal(cloud.label, 'Work bot')
  assert.equal(cloud.cloudName, 'Renamed In Portal')
})

test('cloudLabelIsDerived tells our label apart from a user-typed one', () => {
  assert.equal(cloudLabelIsDerived({ cloudName: 'Research Bot', label: 'Research Bot', url: CLOUD_URL }), true)
  assert.equal(cloudLabelIsDerived({ cloudName: 'Research Bot', label: 'Work bot', url: CLOUD_URL }), false)
  // Registered before names were stored: the host label is ours too.
  assert.equal(cloudLabelIsDerived({ label: 'agent-7f3a.hermes.nousresearch.com', url: CLOUD_URL }), true)
  assert.equal(cloudLabelIsDerived({ label: 'Work bot', url: CLOUD_URL }), false)
})

test('discovery folds a portal rename into the registry label', () => {
  const registry = appliedCloud(emptyRegistry(), 'Research Bot')
  const result = reconcileCloudInstanceNames(registry, [{ dashboardUrl: `${CLOUD_URL}/`, name: 'Renamed In Portal' }])
  const cloud = result.registry.connections.find(connection => connection.kind === 'cloud')

  assert.equal(result.changed, true)
  assert.ok(cloud)
  assert.equal(cloud.label, 'Renamed In Portal')
  assert.equal(cloud.cloudName, 'Renamed In Portal')
})

test('discovery leaves an unchanged roster (and unrelated sources) alone', () => {
  const registry = appliedCloud(emptyRegistry(), 'Research Bot')

  const same = reconcileCloudInstanceNames(registry, [{ dashboardUrl: CLOUD_URL, name: 'Research Bot' }])

  assert.equal(same.changed, false)
  assert.equal(same.registry, registry)

  const unrelated = reconcileCloudInstanceNames(registry, [
    { dashboardUrl: 'https://other.hermes.nousresearch.com', name: 'Other Bot' }
  ])

  assert.equal(unrelated.changed, false)
  assert.equal(unrelated.registry.connections.find(connection => connection.kind === 'cloud')!.label, 'Research Bot')
})

test('discovery keeps a user-typed device name while still tracking the portal name', () => {
  let registry = appliedCloud(emptyRegistry(), 'Research Bot')
  const cloudId = registry.connections.find(connection => connection.kind === 'cloud')!.id

  registry = upsertConnection(registry, {
    ...registry.connections.find(connection => connection.id === cloudId)!,
    label: 'Work bot'
  })

  const result = reconcileCloudInstanceNames(registry, [{ dashboardUrl: CLOUD_URL, name: 'Renamed In Portal' }])
  const cloud = result.registry.connections.find(connection => connection.id === cloudId)

  assert.equal(result.changed, true)
  assert.ok(cloud)
  assert.equal(cloud.label, 'Work bot')
  assert.equal(cloud.cloudName, 'Renamed In Portal')
})

test('discovery suffixes rather than colliding when another source owns the name', () => {
  let registry = appliedCloud(emptyRegistry(), 'Research Bot')

  registry = upsertConnection(registry, {
    id: 'hermes-alex',
    kind: 'remote',
    label: 'Renamed In Portal',
    url: 'https://gateway.example.com',
    authMode: 'oauth'
  })

  const result = reconcileCloudInstanceNames(registry, [{ dashboardUrl: CLOUD_URL, name: 'Renamed In Portal' }])
  const cloud = result.registry.connections.find(connection => connection.kind === 'cloud')

  assert.ok(cloud)
  assert.equal(cloud.label, 'Renamed In Portal 2')
  assert.equal(cloud.cloudName, 'Renamed In Portal')
})

test('a cloud rename survives normalizeRegistry and a label-only edit', () => {
  const registry = appliedCloud(emptyRegistry(), 'Research Bot')
  const reloaded = normalizeRegistry(JSON.parse(JSON.stringify(registry)))
  const cloud = reloaded.connections.find(connection => connection.kind === 'cloud')

  assert.ok(cloud)
  assert.equal(cloud.cloudName, 'Research Bot')

  const edited = normalizeConnectionInput(
    mergeConnectionInput({ id: cloud.id, kind: 'cloud', label: 'Work bot' }, cloud),
    reloaded
  )

  assert.equal(edited.cloudName, 'Research Bot')
  assert.equal(edited.url, CLOUD_URL)
  assert.equal(edited.org, 'nous')
})

test('Apply local moves primary/current to This device without deleting registered remotes', () => {
  const remoteRegistry = reconcileAppliedGlobalConnection(emptyRegistry(), {
    mode: 'remote',
    remote: { url: 'https://one.example.com', authMode: 'oauth' }
  })

  const localRegistry = reconcileAppliedGlobalConnection(remoteRegistry, { mode: 'local', remote: {} })

  assert.equal(localRegistry.primary, LOCAL_CONNECTION_ID)
  assert.equal(localRegistry.lastUsed, LOCAL_CONNECTION_ID)
  assert.equal(localRegistry.connections.filter(connection => connection.kind === 'remote').length, 1)
  assert.equal(resolvedConnectionId(localRegistry, { mode: 'local' }), LOCAL_CONNECTION_ID)
})

test('Apply between two remotes keeps each real registration once and activates the latest', () => {
  const first = reconcileAppliedGlobalConnection(emptyRegistry(), {
    mode: 'remote',
    remote: { url: 'https://one.example.com', authMode: 'oauth' }
  })

  const second = reconcileAppliedGlobalConnection(first, {
    mode: 'remote',
    remote: { url: 'https://two.example.com/', authMode: 'oauth' }
  })

  const remotes = second.connections.filter(connection => connection.kind === 'remote')

  assert.deepEqual(remotes.map(connection => connection.url).sort(), [
    'https://one.example.com',
    'https://two.example.com'
  ])
  assert.equal(new Set(remotes.map(connection => connection.id)).size, 2)
  assert.equal(second.primary, remotes.find(connection => connection.url === 'https://two.example.com')?.id)
  assert.equal(second.lastUsed, second.primary)
})

// --- reconcileRegistryDrift (v1 ↔ v2 healing) ---

test('drift heal registers a v1 remote the registry never learned about and makes it primary', () => {
  // The exact shape users keep reporting: registry migrated while local-only,
  // then Settings → Gateway pointed v1 at a remote. connections.json still
  // says primary 'local', so every launch force-switches off the live remote.
  const drifted = reconcileRegistryDrift(emptyRegistry(), {
    mode: 'remote',
    remote: { url: 'https://agent.example.com:4443', authMode: 'oauth' }
  })

  assert.equal(drifted.changed, true)

  const remote = drifted.registry.connections.find(connection => connection.kind === 'remote')

  assert.ok(remote)
  assert.equal(drifted.registry.primary, remote.id)
  assert.equal(drifted.registry.lastUsed, remote.id)
  // The whole point: the live v1 descriptor can now be named, so the boot pick
  // resolves to the remote instead of re-homing to 'local'. Descriptor shape
  // matches what buildRemoteConnection emits for an oauth remote.
  assert.equal(
    resolvedConnectionId(drifted.registry, {
      authMode: 'oauth',
      baseUrl: 'https://agent.example.com:4443',
      headers: {},
      mode: 'remote',
      remoteKind: 'url'
    }),
    remote.id
  )
})

test('drift heal leaves a registry that already knows the v1 route untouched', () => {
  const registered = reconcileAppliedGlobalConnection(emptyRegistry(), {
    mode: 'remote',
    remote: { url: 'https://agent.example.com', authMode: 'oauth' }
  })

  const drifted = reconcileRegistryDrift(registered, {
    mode: 'remote',
    remote: { url: 'https://AGENT.example.com/', authMode: 'oauth' }
  })

  assert.equal(drifted.changed, false)
  assert.equal(drifted.registry, registered)
})

test('drift heal respects a deliberate primary pick on a registered route', () => {
  // Route IS registered, but the user chose This device in the Connections
  // panel. That is a choice, not drift — never override it.
  let registry = reconcileAppliedGlobalConnection(emptyRegistry(), {
    mode: 'remote',
    remote: { url: 'https://agent.example.com', authMode: 'oauth' }
  })

  registry = setPrimaryConnection(registry, LOCAL_CONNECTION_ID)

  const drifted = reconcileRegistryDrift(registry, {
    mode: 'remote',
    remote: { url: 'https://agent.example.com', authMode: 'oauth' }
  })

  assert.equal(drifted.changed, false)
  assert.equal(drifted.registry.primary, LOCAL_CONNECTION_ID)
})

test('drift heal ignores local and unparseable v1 routes', () => {
  const registry = emptyRegistry()

  for (const v1 of [
    { mode: 'local', remote: {} },
    { mode: 'ssh', remote: {} },
    { mode: 'ssh', remote: { host: '   ' } },
    { mode: 'remote', remote: { url: 'not a url' } },
    { mode: 'remote', remote: {} },
    null
  ]) {
    const drifted = reconcileRegistryDrift(registry, v1)

    assert.equal(drifted.changed, false, `expected no heal for ${JSON.stringify(v1)}`)
    assert.equal(drifted.registry, registry)
  }
})

test('drift heal registers a v1 SSH route the registry never learned about and makes it primary', () => {
  // mgallmur-glitch's shape: registry migrated while local-only, then Settings
  // pointed v1 at an SSH host (host, no url). The registry cannot name it, so
  // primary stays 'local' and the files re-drift after every update relaunch.
  const drifted = reconcileRegistryDrift(emptyRegistry(), {
    mode: 'ssh',
    remote: { host: 'devbox.example.com', user: 'omar', port: 2222 }
  })

  assert.equal(drifted.changed, true)

  const ssh = drifted.registry.connections.find(connection => connection.kind === 'ssh')

  assert.ok(ssh)
  assert.equal(ssh.host, 'devbox.example.com')
  assert.equal(ssh.user, 'omar')
  assert.equal(ssh.port, 2222)
  assert.equal(drifted.registry.primary, ssh.id)
  assert.equal(drifted.registry.lastUsed, ssh.id)
  // The whole point: the live v1 SSH descriptor can now be named.
  assert.equal(
    resolvedConnectionId(drifted.registry, {
      mode: 'remote',
      remoteKind: 'ssh',
      ssh: { host: 'devbox.example.com', user: 'omar', port: 2222 }
    }),
    ssh.id
  )
})

test('drift heal leaves a registry that already knows the v1 SSH route untouched', () => {
  const first = reconcileRegistryDrift(emptyRegistry(), {
    mode: 'ssh',
    remote: { host: 'devbox.example.com', user: 'omar' }
  })

  assert.equal(first.changed, true)

  const drifted = reconcileRegistryDrift(first.registry, {
    mode: 'ssh',
    remote: { host: 'DEVBOX.example.com', user: 'Omar' }
  })

  assert.equal(drifted.changed, false)
  assert.equal(drifted.registry, first.registry)
})

test('drift heal respects a deliberate primary pick on a registered SSH route', () => {
  let registry = reconcileRegistryDrift(emptyRegistry(), {
    mode: 'ssh',
    remote: { host: 'devbox.example.com' }
  }).registry

  registry = setPrimaryConnection(registry, LOCAL_CONNECTION_ID)

  const drifted = reconcileRegistryDrift(registry, {
    mode: 'ssh',
    remote: { host: 'devbox.example.com' }
  })

  assert.equal(drifted.changed, false)
  assert.equal(drifted.registry.primary, LOCAL_CONNECTION_ID)
})

test('drift heal adds the missing SSH source without disturbing other registered sources', () => {
  let registry = emptyRegistry()

  registry = upsertConnection(registry, {
    id: 'homelab',
    kind: 'remote',
    label: 'Homelab',
    url: 'https://homelab.example.com',
    authMode: 'token',
    token: { keep: true }
  })

  const drifted = reconcileRegistryDrift(registry, {
    mode: 'ssh',
    remote: { host: 'devbox.example.com' }
  })

  assert.equal(drifted.changed, true)
  assert.ok(drifted.registry.connections.some(connection => connection.id === 'homelab'))
  assert.ok(drifted.registry.connections.some(connection => connection.kind === 'ssh'))
})

test('drift heal adds the missing remote without disturbing other registered sources', () => {
  let registry = emptyRegistry()

  registry = upsertConnection(registry, {
    id: 'homelab',
    kind: 'remote',
    label: 'Homelab',
    url: 'https://homelab.example.com',
    authMode: 'token',
    token: { keep: true }
  })

  const drifted = reconcileRegistryDrift(registry, {
    mode: 'remote',
    remote: { url: 'https://agent.example.com:4443', authMode: 'oauth' }
  })

  assert.equal(drifted.changed, true)
  assert.equal(drifted.registry.connections.filter(connection => connection.kind === 'remote').length, 2)
  assert.ok(drifted.registry.connections.some(connection => connection.id === 'homelab'))
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

// --- normalizeRegistry per-entry quarantine (#94246 remainder) ---
//
// One malformed entry must never cost the user the rest of the registry, and
// malformed entries are USER DATA: they are preserved under `quarantined`
// (with the raw entry verbatim) instead of being silently deleted on the next
// registry write. "Only deleting connections.json recovers" was the reported
// failure shape; the recovery must never be data loss.

test('normalizeRegistry quarantines malformed entries instead of silently dropping them', () => {
  const registry = normalizeRegistry({
    version: 2,
    primary: 'a',
    connections: [
      { id: 'local', kind: 'local', label: 'This device' },
      { id: 'a', kind: 'remote', label: 'Homelab', url: 'http://10.0.0.5:9119' },
      { id: 'c', kind: 'remote', label: 'No URL entry' },
      { kind: 'nonsense', label: 'Mystery box', extra: 'still my data' },
      { id: 's', kind: 'ssh', label: 'No host ssh' }
    ]
  })

  // Healthy entries all load.
  assert.deepEqual(
    registry.connections.map(c => c.id),
    ['local', 'a']
  )
  assert.equal(registry.primary, 'a')

  // The malformed ones are preserved verbatim, with reasons.
  assert.equal((registry.quarantined || []).length, 3)

  const reasons = registry.quarantined!.map(q => q.reason).sort()

  assert.deepEqual(reasons, ['entry-missing-ssh-host', 'entry-missing-url', 'entry-unrecognized-kind'])

  const mystery = registry.quarantined!.find(q => q.reason === 'entry-unrecognized-kind')

  assert.deepEqual(mystery!.entry, { kind: 'nonsense', label: 'Mystery box', extra: 'still my data' })
})

test('normalizeRegistry preserves previously quarantined entries across round trips', () => {
  const first = normalizeRegistry({
    version: 2,
    connections: [{ id: 'c', kind: 'remote', label: 'No URL entry' }]
  })

  assert.equal((first.quarantined || []).length, 1)

  // Simulate write → read → normalize again (what every registry save does).
  const second = normalizeRegistry(JSON.parse(JSON.stringify(first)))

  assert.equal((second.quarantined || []).length, 1)
  assert.deepEqual(second.quarantined![0].entry, { id: 'c', kind: 'remote', label: 'No URL entry' })
})

test('normalizeRegistry quarantines an entry that explodes during normalization (no whole-load abort)', () => {
  const poisoned: any = { id: 'boom', kind: 'remote', url: 'http://10.0.0.9:9119' }

  Object.defineProperty(poisoned, 'label', {
    enumerable: true,
    get() {
      throw new Error('poisoned entry')
    }
  })

  const registry = normalizeRegistry({
    version: 2,
    primary: 'a',
    connections: [poisoned, { id: 'a', kind: 'remote', label: 'Homelab', url: 'http://10.0.0.5:9119' }]
  })

  // The healthy entry still loads and keeps primary; the poisoned one is
  // quarantined rather than aborting the whole registry load.
  assert.deepEqual(
    registry.connections.filter(c => c.kind === 'remote').map(c => c.id),
    ['a']
  )
  assert.equal(registry.primary, 'a')
  assert.equal((registry.quarantined || []).length, 1)
  assert.equal(registry.quarantined![0].reason, 'entry-normalization-failed')
})

test('normalizeRegistry keeps a clean registry free of the quarantined key and caps quarantine growth', () => {
  const clean = normalizeRegistry({
    version: 2,
    connections: [{ id: 'a', kind: 'remote', label: 'Homelab', url: 'http://10.0.0.5:9119' }]
  })

  assert.equal('quarantined' in clean, false)

  const flooded = normalizeRegistry({
    version: 2,
    connections: Array.from({ length: 100 }, (_, i) => ({ id: `q${i}`, kind: 'remote', label: `No URL ${i}` }))
  })

  assert.ok((flooded.quarantined || []).length <= 20)
})

test('normalizeRegistry quarantines non-object junk items that could still be user data', () => {
  const registry = normalizeRegistry({
    version: 2,
    connections: ['{ mangled json fragment }', null, false, { id: 'a', kind: 'remote', label: 'A', url: 'http://x:1' }]
  })

  assert.deepEqual(
    registry.connections.map(c => c.kind),
    ['local', 'remote']
  )
  // null/false carry no data and are dropped; the string is preserved.
  assert.equal((registry.quarantined || []).length, 1)
  assert.equal(registry.quarantined![0].entry, '{ mangled json fragment }')
})
