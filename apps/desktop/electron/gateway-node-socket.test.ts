import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  httpsOriginFromGatewayWsUrl,
  isAllowedGatewayWsUrl,
  parseGatewayWsUrl,
  shouldDialGatewayFromMain
} from './gateway-node-socket'

test('parseGatewayWsUrl accepts ws(s) /api/ws and rejects everything else', () => {
  assert.equal(parseGatewayWsUrl('wss://gw.example.com/api/ws?ticket=t')?.hostname, 'gw.example.com')
  assert.equal(parseGatewayWsUrl('ws://127.0.0.1:9/api/ws')?.port, '9')
  assert.equal(parseGatewayWsUrl('https://gw.example.com/api/ws'), null)
  assert.equal(parseGatewayWsUrl('wss://gw.example.com/api/status'), null)
  assert.equal(parseGatewayWsUrl('not a url'), null)
})

test('shouldDialGatewayFromMain is only for non-loopback gateway sockets', () => {
  assert.equal(shouldDialGatewayFromMain('wss://hermes.example.com/api/ws?ticket=t'), true)
  assert.equal(shouldDialGatewayFromMain('ws://127.0.0.1:52515/api/ws?token=t'), false)
  assert.equal(shouldDialGatewayFromMain('ws://localhost:9/api/ws'), false)
})

test('isAllowedGatewayWsUrl allowlists saved remote hosts', () => {
  const hosts = ['gw.example.com']

  assert.equal(isAllowedGatewayWsUrl('wss://gw.example.com/api/ws?ticket=a', hosts), true)
  assert.equal(isAllowedGatewayWsUrl('wss://evil.example/api/ws?ticket=a', hosts), false)
  assert.equal(isAllowedGatewayWsUrl('ws://127.0.0.1:9/api/ws', hosts), true)
})

test('httpsOriginFromGatewayWsUrl matches the dashboard scheme/host', () => {
  assert.equal(
    httpsOriginFromGatewayWsUrl('wss://gw.example.com/api/ws?ticket=a'),
    'https://gw.example.com'
  )
  assert.equal(httpsOriginFromGatewayWsUrl('ws://127.0.0.1:9/api/ws'), 'http://127.0.0.1:9')
})
