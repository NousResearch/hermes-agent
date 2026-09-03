import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  chromiumProxyBypassListForGatewayUrls,
  collectRemoteGatewayHosts,
  hostnameFromGatewayUrl,
  proxyBypassRulesForGatewayUrls,
  scutilProxyIsEnabled
} from './gateway-proxy-bypass'

test('hostnameFromGatewayUrl extracts https hosts and drops loopback', () => {
  assert.equal(hostnameFromGatewayUrl('https://gw.example.com'), 'gw.example.com')
  assert.equal(hostnameFromGatewayUrl('https://gw.example.com:9119/chat'), 'gw.example.com')
  assert.equal(hostnameFromGatewayUrl('hermes.example.com'), 'hermes.example.com')
  assert.equal(hostnameFromGatewayUrl('http://127.0.0.1:8642'), null)
  assert.equal(hostnameFromGatewayUrl('http://localhost'), null)
  assert.equal(hostnameFromGatewayUrl(''), null)
  assert.equal(hostnameFromGatewayUrl('ftp://files.example.com'), null)
})

test('collectRemoteGatewayHosts de-dupes and ignores junk', () => {
  assert.deepEqual(
    collectRemoteGatewayHosts([
      'https://gw.example.com',
      'https://GW.example.com/app',
      null,
      'http://127.0.0.1:9',
      'https://other.example.com'
    ]),
    ['gw.example.com', 'other.example.com']
  )
})

test('proxyBypassRulesForGatewayUrls prefixes <local> and returns null when empty', () => {
  assert.equal(proxyBypassRulesForGatewayUrls([]), null)
  assert.equal(proxyBypassRulesForGatewayUrls(['http://localhost']), null)
  assert.equal(
    proxyBypassRulesForGatewayUrls(['https://gw.example.com']),
    '<local>,gw.example.com'
  )
})

test('chromiumProxyBypassListForGatewayUrls is host-only for the command line', () => {
  assert.equal(chromiumProxyBypassListForGatewayUrls([]), null)
  assert.equal(
    chromiumProxyBypassListForGatewayUrls(['https://gw.example.com']),
    'gw.example.com'
  )
})

test('scutilProxyIsEnabled follows Enable flags, not leftover Server/Port', () => {
  assert.equal(
    scutilProxyIsEnabled(`
HTTPEnable : 0
HTTPProxy : 127.0.0.1
HTTPPort : 7892
HTTPSEnable : 0
SOCKSEnable : 0
ProxyAutoConfigEnable : 0
`),
    false
  )
  assert.equal(scutilProxyIsEnabled('HTTPEnable : 1\nHTTPProxy : 127.0.0.1'), true)
  assert.equal(scutilProxyIsEnabled('ProxyAutoConfigEnable : 1'), true)
})
