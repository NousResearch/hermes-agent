import assert from 'node:assert/strict'

import { test } from 'vitest'

import { normalizeRegistry, REGISTRY_VERSION } from './connection-registry'
import { activeRegistrySshScope } from './active-ssh-route'

function registry(lastUsed: string) {
  return normalizeRegistry({
    version: REGISTRY_VERSION,
    primary: 'local',
    lastUsed,
    connections: [
      { id: 'local', kind: 'local', label: 'This device' },
      { id: 'remote', kind: 'remote', label: 'Remote', url: 'https://gateway.test' },
      { id: 'ssh-box', kind: 'ssh', label: 'SSH box', host: 'box.test', user: 'hermes' }
    ]
  })
}

test('returns the v2 SSH pool scope for the last-used registry source', () => {
  assert.equal(activeRegistrySshScope(registry('ssh-box'), 'default'), 'conn:ssh-box::default')
})

test('keeps the active profile in the v2 SSH pool scope', () => {
  assert.equal(activeRegistrySshScope(registry('ssh-box'), 'worker'), 'conn:ssh-box::worker')
})

test('returns null for local and URL-backed registry sources', () => {
  assert.equal(activeRegistrySshScope(registry('local'), 'default'), null)
  assert.equal(activeRegistrySshScope(registry('remote'), 'default'), null)
})
