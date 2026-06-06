/**
 * Regression coverage for the OAuth-session Electron net.request path.
 *
 * Electron net rejects manual Content-Length/Host headers with
 * net::ERR_INVALID_ARGUMENT. Node HTTP helpers may still set Content-Length;
 * this guard is scoped to fetchJsonViaOauthSession only.
 */

const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const source = fs.readFileSync(path.join(__dirname, 'main.cjs'), 'utf8')

function extractFetchJsonViaOauthSession() {
  const start = source.indexOf('function fetchJsonViaOauthSession')
  const end = source.indexOf('// Mint a single-use WS ticket', start)
  assert.notEqual(start, -1, 'fetchJsonViaOauthSession should exist')
  assert.notEqual(end, -1, 'fetchJsonViaOauthSession boundary should exist')
  return source.slice(start, end)
}

test('OAuth Electron net request does not set forbidden Content-Length header', () => {
  const fn = extractFetchJsonViaOauthSession()

  assert.match(fn, /electronNet\.request/)
  assert.doesNotMatch(fn, /setHeader\(['"]Content-Length['"]/)
  assert.match(fn, /request\.write\(body\)/)
})
