import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  CF_ACCESS_CLIENT_ID_HEADER,
  CF_ACCESS_CLIENT_SECRET_HEADER,
  cloudflareAccessHeaders,
  CloudflareAccessWebSocketHeaderRegistry,
  mergeCloudflareAccessHeaders,
  requestMatchesRemoteBase
} from './cloudflare-access'

const ACCESS_HEADERS = cloudflareAccessHeaders('client.access', 'secret-value')

test('cloudflareAccessHeaders returns the standard pair or nothing', () => {
  assert.deepEqual(ACCESS_HEADERS, {
    [CF_ACCESS_CLIENT_ID_HEADER]: 'client.access',
    [CF_ACCESS_CLIENT_SECRET_HEADER]: 'secret-value'
  })
  assert.deepEqual(cloudflareAccessHeaders('', ''), {})
})

test('cloudflareAccessHeaders rejects incomplete credentials', () => {
  assert.throws(() => cloudflareAccessHeaders('client.access', ''), /provided together/)
  assert.throws(() => cloudflareAccessHeaders('', 'secret-value'), /provided together/)
})

test('requestMatchesRemoteBase covers HTTP and WebSocket requests without leaking to sibling paths', () => {
  assert.equal(
    requestMatchesRemoteBase('https://gateway.example.com/hermes/api/status', 'https://gateway.example.com/hermes'),
    true
  )
  assert.equal(
    requestMatchesRemoteBase('wss://gateway.example.com/hermes/api/ws?token=x', 'https://gateway.example.com/hermes'),
    true
  )
  assert.equal(
    requestMatchesRemoteBase(
      'https://gateway.example.com/hermes-other/api/status',
      'https://gateway.example.com/hermes'
    ),
    false
  )
  assert.equal(
    requestMatchesRemoteBase('https://other.example.com/hermes/api/status', 'https://gateway.example.com/hermes'),
    false
  )
  assert.equal(
    requestMatchesRemoteBase('http://gateway.example.com/hermes/api/status', 'https://gateway.example.com/hermes'),
    false
  )
})

test('mergeCloudflareAccessHeaders replaces differently-cased stale values', () => {
  assert.deepEqual(
    mergeCloudflareAccessHeaders(
      {
        Accept: 'application/json',
        'cf-access-client-id': 'stale-id',
        'CF-ACCESS-CLIENT-SECRET': 'stale-secret'
      },
      ACCESS_HEADERS
    ),
    {
      Accept: 'application/json',
      [CF_ACCESS_CLIENT_ID_HEADER]: 'client.access',
      [CF_ACCESS_CLIENT_SECRET_HEADER]: 'secret-value'
    }
  )
})

test('WebSocket header registry scopes credentials to an exact, short-lived URL', () => {
  const registry = new CloudflareAccessWebSocketHeaderRegistry(1_000)
  const url = 'wss://gateway.example.com/api/ws?ticket=one-time'

  registry.remember(url, ACCESS_HEADERS, 10_000)

  assert.deepEqual(registry.resolve(url, 10_500), ACCESS_HEADERS)
  assert.deepEqual(registry.resolve('wss://gateway.example.com/api/ws?ticket=other', 10_500), {})
  assert.deepEqual(registry.resolve(url, 11_000), {})
})
