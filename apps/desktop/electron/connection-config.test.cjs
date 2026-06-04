/**
 * Tests for electron/connection-config.cjs.
 *
 * Run with: node --test electron/connection-config.test.cjs
 * (Wire into npm test:desktop:platforms in package.json.)
 *
 * These are the pure helpers behind the remote-gateway connection settings:
 * URL normalization, WS-URL construction (token vs OAuth ticket), auth-mode
 * classification from /api/status, the coerce-time auth-mode resolution rules,
 * and the OAuth session-cookie detector.
 */

const test = require('node:test')
const assert = require('node:assert/strict')

const {
  AT_COOKIE_VARIANTS,
  authModeFromStatus,
  buildGatewayWsUrl,
  buildGatewayWsUrlWithTicket,
  cookiesHaveSession,
  normalizeRemoteBaseUrl,
  resolveAuthMode,
  tokenPreview
} = require('./connection-config.cjs')

// --- normalizeRemoteBaseUrl ---

test('normalizeRemoteBaseUrl strips trailing slashes, hash, and query', () => {
  assert.equal(normalizeRemoteBaseUrl('https://gw.example.com/'), 'https://gw.example.com')
  assert.equal(normalizeRemoteBaseUrl('https://gw.example.com/hermes/'), 'https://gw.example.com/hermes')
  assert.equal(normalizeRemoteBaseUrl('https://gw.example.com/hermes?x=1#frag'), 'https://gw.example.com/hermes')
})

test('normalizeRemoteBaseUrl preserves a path prefix', () => {
  assert.equal(normalizeRemoteBaseUrl('https://host/hermes'), 'https://host/hermes')
})

test('normalizeRemoteBaseUrl rejects empty input', () => {
  assert.throws(() => normalizeRemoteBaseUrl(''), /required/)
  assert.throws(() => normalizeRemoteBaseUrl('   '), /required/)
})

test('normalizeRemoteBaseUrl rejects non-http(s) protocols', () => {
  assert.throws(() => normalizeRemoteBaseUrl('ftp://host'), /http:\/\/ or https:\/\//)
  assert.throws(() => normalizeRemoteBaseUrl('file:///etc/passwd'), /http:\/\/ or https:\/\//)
})

test('normalizeRemoteBaseUrl rejects garbage', () => {
  assert.throws(() => normalizeRemoteBaseUrl('not a url'), /not valid/)
})

// --- buildGatewayWsUrl (token) ---

test('buildGatewayWsUrl uses wss for https and bakes the token', () => {
  assert.equal(
    buildGatewayWsUrl('https://gw.example.com', 'tok123'),
    'wss://gw.example.com/api/ws?token=tok123'
  )
})

test('buildGatewayWsUrl uses ws for http', () => {
  assert.equal(
    buildGatewayWsUrl('http://127.0.0.1:9119', 'abc'),
    'ws://127.0.0.1:9119/api/ws?token=abc'
  )
})

test('buildGatewayWsUrl honors a path prefix', () => {
  assert.equal(
    buildGatewayWsUrl('https://host/hermes', 't'),
    'wss://host/hermes/api/ws?token=t'
  )
})

test('buildGatewayWsUrl url-encodes the token', () => {
  assert.equal(
    buildGatewayWsUrl('https://host', 'a/b c+d'),
    'wss://host/api/ws?token=a%2Fb%20c%2Bd'
  )
})

// --- buildGatewayWsUrlWithTicket (oauth) ---

test('buildGatewayWsUrlWithTicket uses ?ticket= not ?token=', () => {
  const url = buildGatewayWsUrlWithTicket('https://gw.example.com/hermes', 'tkt-9')
  assert.equal(url, 'wss://gw.example.com/hermes/api/ws?ticket=tkt-9')
  assert.ok(!url.includes('token='))
})

test('buildGatewayWsUrlWithTicket url-encodes the ticket', () => {
  assert.equal(
    buildGatewayWsUrlWithTicket('https://host', 'a+b/c'),
    'wss://host/api/ws?ticket=a%2Bb%2Fc'
  )
})

// --- authModeFromStatus ---

test('authModeFromStatus returns oauth when auth_required is true', () => {
  assert.equal(authModeFromStatus({ auth_required: true, auth_providers: ['nous'] }), 'oauth')
})

test('authModeFromStatus returns token when auth_required is false/missing', () => {
  assert.equal(authModeFromStatus({ auth_required: false }), 'token')
  assert.equal(authModeFromStatus({}), 'token')
  assert.equal(authModeFromStatus(null), 'token')
  assert.equal(authModeFromStatus(undefined), 'token')
})

// --- resolveAuthMode ---

test('resolveAuthMode: explicit input wins over existing', () => {
  assert.equal(resolveAuthMode('oauth', 'token'), 'oauth')
  assert.equal(resolveAuthMode('token', 'oauth'), 'token')
})

test('resolveAuthMode: falls back to existing when input absent', () => {
  assert.equal(resolveAuthMode(undefined, 'oauth'), 'oauth')
  assert.equal(resolveAuthMode(undefined, 'token'), 'token')
  assert.equal(resolveAuthMode('', 'oauth'), 'oauth')
})

test('resolveAuthMode: defaults to token when nothing is set', () => {
  assert.equal(resolveAuthMode(undefined, undefined), 'token')
  assert.equal(resolveAuthMode(null, null), 'token')
})

test('resolveAuthMode: ignores unknown values, defaults to token', () => {
  assert.equal(resolveAuthMode('bogus', 'also-bogus'), 'token')
})

// --- cookiesHaveSession ---

test('cookiesHaveSession detects the bare access-token cookie', () => {
  assert.equal(cookiesHaveSession([{ name: 'hermes_session_at', value: 'x' }]), true)
})

test('cookiesHaveSession detects the __Host- and __Secure- prefixed variants', () => {
  assert.equal(cookiesHaveSession([{ name: '__Host-hermes_session_at', value: 'x' }]), true)
  assert.equal(cookiesHaveSession([{ name: '__Secure-hermes_session_at', value: 'x' }]), true)
})

test('cookiesHaveSession is false for an empty value', () => {
  assert.equal(cookiesHaveSession([{ name: 'hermes_session_at', value: '' }]), false)
})

test('cookiesHaveSession ignores unrelated cookies', () => {
  assert.equal(cookiesHaveSession([{ name: 'hermes_session_rt', value: 'x' }]), false)
  assert.equal(cookiesHaveSession([{ name: 'other', value: 'x' }]), false)
})

test('cookiesHaveSession handles non-arrays', () => {
  assert.equal(cookiesHaveSession(null), false)
  assert.equal(cookiesHaveSession(undefined), false)
  assert.equal(cookiesHaveSession([]), false)
})

test('AT_COOKIE_VARIANTS covers all three deploy shapes', () => {
  assert.deepEqual(AT_COOKIE_VARIANTS, [
    '__Host-hermes_session_at',
    '__Secure-hermes_session_at',
    'hermes_session_at'
  ])
})

// --- tokenPreview ---

test('tokenPreview returns null for empty', () => {
  assert.equal(tokenPreview(''), null)
  assert.equal(tokenPreview(null), null)
})

test('tokenPreview returns set for short tokens', () => {
  assert.equal(tokenPreview('12345678'), 'set')
})

test('tokenPreview returns a masked suffix for long tokens', () => {
  assert.equal(tokenPreview('abcdefghijklmnop'), '...klmnop')
})
