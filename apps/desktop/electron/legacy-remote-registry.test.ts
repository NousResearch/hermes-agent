import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  migrateV1ToRegistry,
  normalizeRegistry,
  reconcileAppliedGlobalConnection,
  reconcileRegistryDrift
} from './connection-registry'
import { savedLegacyTunnelConnectionId } from './legacy-remote-config'
import { LEGACY_OAUTH_PARTITION, resolveOauthPartition } from './oauth-partition'

test('v1 tunnel apply distinguishes equal loopback URLs by SSH target', () => {
  const direct = reconcileAppliedGlobalConnection(normalizeRegistry(null), {
    mode: 'remote',
    remote: { url: 'http://127.0.0.1:9119', authMode: 'oauth' }
  })

  const tunnelled = reconcileAppliedGlobalConnection(direct, {
    mode: 'remote',
    remote: { url: 'http://127.0.0.1:9119', authMode: 'oauth', sshTunnel: { host: 'alice@gw-host' } }
  })

  assert.notEqual(tunnelled.primary, direct.primary)
  assert.deepEqual(tunnelled.connections.find(c => c.id === tunnelled.primary)?.sshTunnel, {
    host: 'alice@gw-host'
  })
  assert.equal(tunnelled.connections.filter(c => c.kind === 'remote').length, 2)
})

test('v1 migration and drift preserve tunnel identity', () => {
  const config = {
    mode: 'remote',
    remote: { url: 'http://127.0.0.1:9119', authMode: 'oauth', sshTunnel: { host: 'alice@gw-host' } }
  }

  const migrated = migrateV1ToRegistry(config)
  assert.deepEqual(migrated.connections.find(c => c.id === migrated.primary)?.sshTunnel, { host: 'alice@gw-host' })

  const direct = reconcileAppliedGlobalConnection(normalizeRegistry(null), {
    mode: 'remote',
    remote: { url: 'http://127.0.0.1:9119', authMode: 'oauth' }
  })

  const healed = reconcileRegistryDrift(direct, config)
  assert.equal(healed.changed, true)
  assert.deepEqual(healed.registry.connections.find(c => c.id === healed.registry.primary)?.sshTunnel, {
    host: 'alice@gw-host'
  })
})

// Mirrors main.ts legacyRemoteDraftSource → oauth-login → apply: the draft id is the id the save keeps.
function draftId(registry: ReturnType<typeof normalizeRegistry>, host: string): string {
  const remote = { url: 'http://127.0.0.1:9119', authMode: 'oauth', sshTunnel: { host } }
  const draft = reconcileAppliedGlobalConnection(registry, { mode: 'remote', remote })

  return draft.primary
}

test('a tunnelled v1 draft signs in on the jar its saved connection reads', () => {
  const registry = normalizeRegistry(null)
  const id = draftId(registry, 'alice@gw-host')
  const localUrl = 'http://127.0.0.1:41733'

  const loginJar = resolveOauthPartition(localUrl, {
    connectionId: id,
    pendingAuthMode: 'oauth',
    pendingKind: 'remote',
    registry
  })

  const remote = { url: 'http://127.0.0.1:9119', authMode: 'oauth', sshTunnel: { host: 'alice@gw-host' } }
  const saved = reconcileAppliedGlobalConnection(registry, { mode: 'remote', remote })

  assert.equal(saved.primary, id)
  assert.notEqual(loginJar, LEGACY_OAUTH_PARTITION)
  assert.equal(
    resolveOauthPartition('http://127.0.0.1:40999', { connectionId: id, registry: saved, v1RemoteUrl: remote.url }),
    loginJar
  )
})

test('two unsaved v1 drafts with the same URL and different SSH hosts get separate jars', () => {
  const registry = normalizeRegistry(null)

  const first = reconcileAppliedGlobalConnection(registry, {
    mode: 'remote',
    remote: { url: 'http://127.0.0.1:9119', authMode: 'oauth', sshTunnel: { host: 'alice@box-a' } }
  })

  const secondId = draftId(first, 'alice@box-b')

  const jar = (id: string) =>
    resolveOauthPartition('http://127.0.0.1:41733', {
      connectionId: id,
      pendingAuthMode: 'oauth',
      pendingKind: 'remote',
      registry: first
    })

  assert.notEqual(secondId, first.primary)
  assert.notEqual(jar(secondId), jar(first.primary))
})

test('URL-only sign-out and status calls resolve the saved tunnelled v1 jar', () => {
  const remote = { url: 'http://127.0.0.1:9119', authMode: 'oauth', sshTunnel: { host: 'alice@gw-host' } }
  const config = { mode: 'remote', remote }
  const registry = reconcileAppliedGlobalConnection(normalizeRegistry(null), config)
  const id = savedLegacyTunnelConnectionId('http://127.0.0.1:9119/', config, registry)

  assert.equal(id, registry.primary)
  assert.equal(
    resolveOauthPartition(remote.url, { connectionId: id, registry, v1RemoteUrl: remote.url }),
    resolveOauthPartition('http://127.0.0.1:41733', { connectionId: registry.primary, registry })
  )
  assert.equal(savedLegacyTunnelConnectionId('http://127.0.0.1:9120', config, registry), '')
  assert.equal(savedLegacyTunnelConnectionId(remote.url, { mode: 'remote', remote: { url: remote.url } }, registry), '')
  assert.equal(savedLegacyTunnelConnectionId('not a url', config, registry), '')
  assert.equal(resolveOauthPartition(remote.url, { registry, v1RemoteUrl: remote.url }), LEGACY_OAUTH_PARTITION)
})
