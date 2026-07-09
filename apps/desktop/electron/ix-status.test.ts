/**
 * Tests for electron/ix-status.ts — update manifest polling, the VPN
 * deep-status combiner (handshake + exit-IP), the admin-mcp lamp and the
 * Cognito/Hermes-init helpers.
 *
 * Run with: node --test electron/ix-status.test.ts
 * (Wired into npm test:desktop:platforms in package.json.)
 *
 * Why this matters: the lamps must never show a fake state — VPN green
 * requires REAL egress through the company exit IP, MCP green requires an
 * AUTHENTICATED tools/list, and "update available" requires a manifest
 * version strictly newer than the running app.
 */

import assert from 'node:assert/strict'
import test from 'node:test'

import {
  combineVpnStatus,
  compareVersions,
  fetchMcpLampStatus,
  looksLikeWireGuardConf,
  minimalHermesConfigYaml,
  parseUpdateManifest,
  updatePlatformKey,
  upsertEnvLine,
  wgHandshakeAgeSecs
} from './ix-status'

/* ── update manifest ─────────────────────────────────────────────────────── */

test('compareVersions orders dotted versions (v-prefix tolerated)', () => {
  assert.ok(compareVersions('0.18.0', '0.17.0') > 0)
  assert.ok(compareVersions('v1.0.0', '0.99.9') > 0)
  assert.equal(compareVersions('0.17.0', '0.17.0'), 0)
  assert.ok(compareVersions('0.17.0', '0.17.1') < 0)
  // numeric, not lexicographic
  assert.ok(compareVersions('0.10.0', '0.9.0') > 0)
})

test('parseUpdateManifest: newer version flips updateAvailable, platform url wins', () => {
  const key = updatePlatformKey('darwin', 'arm64')

  assert.equal(key, 'darwin-aarch64')

  const status = parseUpdateManifest(
    {
      version: '0.18.0',
      notes: 'fixes',
      url: 'https://example.com/releases',
      platforms: { 'darwin-aarch64': { url: 'https://example.com/mac-arm.dmg' } }
    },
    '0.17.0',
    key
  )

  assert.equal(status.updateAvailable, true)
  assert.equal(status.url, 'https://example.com/mac-arm.dmg')
  assert.equal(status.latestVersion, '0.18.0')
})

test('parseUpdateManifest: same/older/missing versions are NOT updates', () => {
  assert.equal(parseUpdateManifest({ version: '0.17.0' }, '0.17.0').updateAvailable, false)
  assert.equal(parseUpdateManifest({ version: '0.16.9' }, '0.17.0').updateAvailable, false)
  assert.equal(parseUpdateManifest({}, '0.17.0').updateAvailable, false)
  assert.match(parseUpdateManifest({}, '0.17.0').detail, /no version/)
})

/* ── VPN deep status ─────────────────────────────────────────────────────── */

const EXIT = '3.224.15.124'

test('vpn: green ONLY when tunnel up + egress via exit + fresh handshake', () => {
  const status = combineVpnStatus({
    tunnelUp: true,
    handshakeAgeSecs: 12,
    egressIp: EXIT,
    expectedExitIp: EXIT
  })

  assert.equal(status.state, 'connected')
})

test('vpn: tunnel down is disconnected regardless of anything else', () => {
  const status = combineVpnStatus({
    tunnelUp: false,
    handshakeAgeSecs: 5,
    egressIp: EXIT,
    expectedExitIp: EXIT
  })

  assert.equal(status.state, 'disconnected')
})

test('vpn: egress NOT via the exit IP is degraded, never green', () => {
  const status = combineVpnStatus({
    tunnelUp: true,
    handshakeAgeSecs: 5,
    egressIp: '203.0.113.7',
    expectedExitIp: EXIT
  })

  assert.equal(status.state, 'degraded')
  assert.match(status.detail, /203\.0\.113\.7/)
})

test('vpn: unreadable handshake (root-only wg show) still counts when egress matches', () => {
  const status = combineVpnStatus({
    tunnelUp: true,
    handshakeAgeSecs: null,
    egressIp: EXIT,
    expectedExitIp: EXIT
  })

  assert.equal(status.state, 'connected')
  assert.match(status.detail, /handshake unreadable/)
})

test('vpn: stale handshake with matching egress is degraded', () => {
  const status = combineVpnStatus({
    tunnelUp: true,
    handshakeAgeSecs: 60 * 60,
    egressIp: EXIT,
    expectedExitIp: EXIT
  })

  assert.equal(status.state, 'degraded')
})

test('vpn: failed exit-IP check is an error state (no fake green)', () => {
  const status = combineVpnStatus({
    tunnelUp: true,
    handshakeAgeSecs: 5,
    egressIp: null,
    egressError: 'timeout',
    expectedExitIp: EXIT
  })

  assert.equal(status.state, 'error')
  assert.match(status.detail, /timeout/)
})

test('wgHandshakeAgeSecs parses epoch output and treats 0 as no handshake', () => {
  const now = Math.floor(Date.now() / 1000)
  const age = wgHandshakeAgeSecs('usa-vpn', () => `peerkey\t${now - 30}\n`)

  assert.ok(age !== null && age >= 30 && age < 40)
  assert.equal(
    wgHandshakeAgeSecs('usa-vpn', () => 'peerkey\t0\n'),
    null
  )
  assert.equal(
    wgHandshakeAgeSecs('usa-vpn', () => {
      throw new Error('Operation not permitted')
    }),
    null
  )
})

test('looksLikeWireGuardConf requires Interface, Peer and PrivateKey', () => {
  assert.equal(looksLikeWireGuardConf('[Interface]\nPrivateKey = x\n[Peer]\nEndpoint = 1.2.3.4:51820'), true)
  assert.equal(looksLikeWireGuardConf('just some text'), false)
})

/* ── MCP lamp ────────────────────────────────────────────────────────────── */

function fakeFetch(routes: Record<string, { status: number; body?: unknown }>) {
  return (async (url: RequestInfo | URL) => {
    const key = Object.keys(routes).find(route => String(url).includes(route))
    const route = key ? routes[key] : { status: 404 }

    return {
      ok: route.status >= 200 && route.status < 300,
      status: route.status,
      json: async () => route.body ?? {},
      text: async () => JSON.stringify(route.body ?? {})
    } as Response
  }) as typeof fetch
}

test('mcp lamp: green requires reachable /healthz AND authenticated tools/list', async () => {
  const status = await fetchMcpLampStatus(
    'https://gw.example.com/',
    'tok',
    fakeFetch({
      '/healthz': { status: 200 },
      'gw.example.com': { status: 200, body: { result: { tools: [{ name: 'a' }, { name: 'b' }] } } }
    })
  )

  assert.equal(status.state, 'green')
  assert.equal(status.toolCount, 2)
})

test('mcp lamp: rejected token is grey, not red', async () => {
  const status = await fetchMcpLampStatus(
    'https://gw.example.com/',
    'bad',
    fakeFetch({ '/healthz': { status: 200 }, 'gw.example.com': { status: 401 } })
  )

  assert.equal(status.state, 'grey')
  assert.equal(status.authenticated, false)
})

test('mcp lamp: no token is grey with guidance; unreachable is red', async () => {
  const noToken = await fetchMcpLampStatus('https://gw.example.com/', '', fakeFetch({ '/healthz': { status: 200 } }))

  assert.equal(noToken.state, 'grey')

  const down = await fetchMcpLampStatus('https://gw.example.com/', 'tok', fakeFetch({ '/healthz': { status: 502 } }))

  assert.equal(down.state, 'red')
})

/* ── Hermes init helpers ─────────────────────────────────────────────────── */

test('minimalHermesConfigYaml wires the LiteLLM gateway as model provider', () => {
  const yaml = minimalHermesConfigYaml('https://litellm.intelli-verse-x.ai', 'https://admin-mcp.intelli-verse-x.ai/')

  assert.match(yaml, /base_url: "https:\/\/litellm\.intelli-verse-x\.ai\/v1"/)
  assert.match(yaml, /admin-mcp/)
  // trailing-slash / pre-versioned URLs don't double up
  assert.match(minimalHermesConfigYaml('https://x.ai/v1', 'g'), /base_url: "https:\/\/x\.ai\/v1"/)
})

test('upsertEnvLine replaces existing keys and appends new ones', () => {
  const first = upsertEnvLine('', 'ADMIN_MCP_TOKEN', 'abc')

  assert.equal(first, 'ADMIN_MCP_TOKEN=abc\n')

  const replaced = upsertEnvLine('ADMIN_MCP_TOKEN=old\nOTHER=1\n', 'ADMIN_MCP_TOKEN', 'new')

  assert.match(replaced, /ADMIN_MCP_TOKEN=new/)
  assert.doesNotMatch(replaced, /old/)
  assert.match(replaced, /OTHER=1/)
})
