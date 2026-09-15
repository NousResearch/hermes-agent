import assert from 'node:assert/strict'

import { test } from 'vitest'

import { normalizeRegistry, REGISTRY_VERSION } from './connection-registry'
import { headersForRemoteRequest } from './remote-request-headers'

const accessHeaders = {
  'CF-Access-Client-Id': { encoding: 'plain', value: 'client-id' },
  'CF-Access-Client-Secret': { encoding: 'plain', value: 'client-secret' }
}

function registry(primary: string, connections: Record<string, unknown>[]) {
  return normalizeRegistry({
    version: REGISTRY_VERSION,
    primary,
    connections: [{ id: 'local', kind: 'local', label: 'This device' }, ...connections]
  })
}

test('uses registry-primary headers for matching HTTP requests when v1 mode is local', () => {
  const headers = headersForRemoteRequest(
    'https://gateway.example/api/profiles/sessions/sidebar?recents_profile=default',
    { mode: 'local' },
    registry('gateway', [{ id: 'gateway', kind: 'remote', label: 'Gateway', url: 'https://gateway.example', headers: accessHeaders }])
  )

  assert.deepEqual(headers, accessHeaders)
})

test('does not leak registry-primary headers to a different gateway', () => {
  const headers = headersForRemoteRequest(
    'https://other.example/api/config',
    { mode: 'local' },
    registry('gateway', [{ id: 'gateway', kind: 'remote', label: 'Gateway', url: 'https://gateway.example', headers: accessHeaders }])
  )

  assert.deepEqual(headers, {})
})

test('keeps v1 remote settings authoritative over a registry-primary gateway', () => {
  const headers = headersForRemoteRequest(
    'https://gateway.example/api/config',
    { mode: 'remote', remote: { url: 'https://gateway.example', headers: { 'X-Settings': 'settings' } } },
    registry('gateway', [{ id: 'gateway', kind: 'remote', label: 'Gateway', url: 'https://gateway.example', headers: accessHeaders }])
  )

  assert.deepEqual(headers, { 'X-Settings': 'settings' })
})
