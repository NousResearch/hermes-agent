/**
 * Tests for electron/connection-registry.ts registry state: input validation,
 * normalization, migration, reconciliation, and persisted dial fields.
 */

import assert from 'node:assert/strict'

import { test } from 'vitest'

import type { ConnectionRegistry } from './connection-registry'
import {
  connectionDialFieldsChanged,
  labelKey,
  LOCAL_CONNECTION_ID,
  mergeConnectionInput,
  migrateV1ToRegistry,
  normalizeConnectionInput,
  normalizeRegistry,
  reconcileAppliedGlobalConnection,
  reconcileRegistryDrift,
  REGISTRY_VERSION,
  removeConnection,
  resolvedConnectionId,
  setConnectionLaunchMode,
  setLastUsedConnection,
  setPrimaryConnection,
  updateEligibility,
  upsertConnection
} from './connection-registry'

function emptyRegistry(): ConnectionRegistry {
  return normalizeRegistry(null)
}

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

test('an ssh entry keeps its session token through a label rename', () => {
  // #103795, second half: saveRegistryConnection resolves the surviving
  // envelope (resolvePersistedRemoteToken keeps the stored one when the
  // editor sends no new value) and hands it to normalizeConnectionInput —
  // whose ssh branch used to drop it, so renaming a connection wiped the live
  // backend's reuse credential and re-armed the reap-and-respawn loop.
  const stored = {
    host: 'spark1',
    id: 'spark',
    kind: 'ssh' as const,
    label: 'Spark',
    port: 2222,
    token: { enc: 'ssh-session-token' },
    user: 'tek'
  }

  const merged = mergeConnectionInput(
    { id: 'spark', kind: 'ssh', label: 'Spark (office)', token: stored.token },
    stored
  )

  const renamed = normalizeConnectionInput(merged, emptyRegistry())

  assert.equal(renamed.label, 'Spark (office)')
  assert.deepEqual(renamed.token, { enc: 'ssh-session-token' })
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

test('normalizeRegistry keeps the persisted ssh session token across a cold read', () => {
  // #103795: persistSshConnectionToken() writes the adopted per-serve token
  // onto the ssh entry, but normalization rebuilt the entry from the DIAL
  // fields alone and dropped it. The token then lived only in the mtime-keyed
  // in-process cache, so the next launch dialed with an empty reuseToken,
  // failed remote-lifecycle's `Boolean(reuseToken)` reuse gate, reaped a
  // healthy owned backend and respawned it on a new port — while the renderer
  // kept dialing the old token and got 403 forever.
  const saved = {
    version: REGISTRY_VERSION,
    primary: 'spark',
    connections: [
      { id: LOCAL_CONNECTION_ID, kind: 'local', label: 'This device' },
      {
        id: 'spark',
        kind: 'ssh',
        label: 'Spark',
        host: 'spark1',
        user: 'tek',
        port: 2222,
        token: { enc: 'ssh-session-token' }
      }
    ]
  }

  const registry = normalizeRegistry(saved)
  const spark = registry.connections.find(connection => connection.id === 'spark')

  assert.deepEqual(spark?.token, { enc: 'ssh-session-token' })
  assert.equal(spark?.host, 'spark1')

  // Write → read → normalize again: the token must survive every cold read,
  // not just the first.
  const reread = normalizeRegistry(JSON.parse(JSON.stringify(registry)))

  assert.deepEqual(reread.connections.find(connection => connection.id === 'spark')?.token, {
    enc: 'ssh-session-token'
  })
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
