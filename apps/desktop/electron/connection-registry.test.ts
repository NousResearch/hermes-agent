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
  connectionIdForLabel,
  labelKey,
  labelSlug,
  LOCAL_CONNECTION_ID,
  mergeConnectionInput,
  migrateV1ToRegistry,
  normalizeConnectionInput,
  normalizeRegistry,
  REGISTRY_VERSION,
  removeConnection,
  setPrimaryConnection,
  uniqueLabel,
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
