/**
 * Tests for electron/gateway-ws-origin.ts.
 *
 * The pure rewrite decision behind the renderer's gateway-WS same-origin
 * handshake: which URLs count as gateway WS upgrades, what Origin value they
 * get, and that everything else passes through untouched.
 */

import assert from 'node:assert/strict'

import { test } from 'vitest'

import { applyGatewayWsOrigin, gatewayWsOrigin } from './gateway-ws-origin'

// --- gatewayWsOrigin: matching ---

test('gatewayWsOrigin maps gateway WS endpoints to their own web origin', () => {
  assert.equal(gatewayWsOrigin('ws://100.64.0.7:9119/api/ws?ticket=abc'), 'http://100.64.0.7:9119')
  assert.equal(gatewayWsOrigin('ws://100.64.0.7:9119/api/pub?channel=c1'), 'http://100.64.0.7:9119')
  assert.equal(gatewayWsOrigin('ws://100.64.0.7:9119/api/events?channel=c1'), 'http://100.64.0.7:9119')
  assert.equal(gatewayWsOrigin('wss://gw.example.com/api/ws?ticket=abc'), 'https://gw.example.com')
})

test('gatewayWsOrigin supports path prefixes and trailing slashes', () => {
  assert.equal(gatewayWsOrigin('wss://box.tailnet.ts.net/hermes/api/ws?ticket=t'), 'https://box.tailnet.ts.net')
  assert.equal(gatewayWsOrigin('ws://127.0.0.1:9119/api/ws/'), 'http://127.0.0.1:9119')
})

test('gatewayWsOrigin ignores non-WS schemes and non-gateway paths', () => {
  assert.equal(gatewayWsOrigin('http://100.64.0.7:9119/api/ws'), null)
  assert.equal(gatewayWsOrigin('ws://100.64.0.7:9119/api/pty'), null)
  assert.equal(gatewayWsOrigin('ws://100.64.0.7:9119/api/wsx'), null)
  assert.equal(gatewayWsOrigin('ws://example.com/chat'), null)
  assert.equal(gatewayWsOrigin('not a url'), null)
  assert.equal(gatewayWsOrigin(''), null)
  assert.equal(gatewayWsOrigin(null), null)
})

// --- applyGatewayWsOrigin: header rewrite ---

test('applyGatewayWsOrigin replaces the Origin header for gateway upgrades', () => {
  const headers = applyGatewayWsOrigin('ws://100.64.0.7:9119/api/ws?ticket=t', {
    Origin: 'http://127.0.0.1:5174',
    'User-Agent': 'test'
  })

  assert.deepEqual(headers, { Origin: 'http://100.64.0.7:9119', 'User-Agent': 'test' })
})

test('applyGatewayWsOrigin removes any casing of the original Origin key', () => {
  const headers = applyGatewayWsOrigin('ws://100.64.0.7:9119/api/ws', { origin: 'http://127.0.0.1:5174' })

  assert.deepEqual(headers, { Origin: 'http://100.64.0.7:9119' })
})

test('applyGatewayWsOrigin stamps Origin even when the handshake carried none', () => {
  const headers = applyGatewayWsOrigin('ws://100.64.0.7:9119/api/ws', {})

  assert.deepEqual(headers, { Origin: 'http://100.64.0.7:9119' })
})

test('applyGatewayWsOrigin returns null for non-gateway requests (pass through)', () => {
  assert.equal(applyGatewayWsOrigin('wss://example.com/socket', { Origin: 'http://127.0.0.1:5174' }), null)
  assert.equal(applyGatewayWsOrigin('https://gw.example.com/api/ws', { Origin: 'http://127.0.0.1:5174' }), null)
})

test('applyGatewayWsOrigin does not mutate the input headers', () => {
  const input = { Origin: 'http://127.0.0.1:5174' }

  applyGatewayWsOrigin('ws://100.64.0.7:9119/api/ws', input)

  assert.deepEqual(input, { Origin: 'http://127.0.0.1:5174' })
})
