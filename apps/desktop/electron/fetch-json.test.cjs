/**
 * Tests for apps/desktop/electron/fetch-json.cjs — covers the header
 * contract that surfaces as a hard user error in password-auth mode
 * (#63707).
 *
 * Run with: node --test apps/desktop/electron/fetch-json.test.cjs
 *
 * The bug: before this PR, fetchJson set 'X-Hermes-Session-Token' to whatever
 * value the caller passed (including undefined), and Node's http.request
 * chokes on an undefined header value with ERR_INVALID_ARG_TYPE before
 * anything hits the network. For password-auth remotes, `token` is
 * legitimately undefined (auth is via HttpOnly cookie, not token header),
 * so every REST mutation surfaced as `net::ERR_INVALID_ARGUMENT` to the
 * renderer.
 *
 * The fix: extract fetchJsonHeaders() so we can test the header contract
 * directly. The header is now omitted entirely when token is nullish.
 */

const test = require('node:test')
const assert = require('node:assert/strict')

const { fetchJsonHeaders } = require('./fetch-json.cjs')

// --- fetchJsonHeaders: the contract ---

test('fetchJsonHeaders sets Content-Type always', () => {
  const headers = fetchJsonHeaders(undefined, undefined)
  assert.equal(headers['Content-Type'], 'application/json')
})

test('fetchJsonHeaders sets X-Hermes-Session-Token when token is a string', () => {
  const headers = fetchJsonHeaders('my-token', undefined)
  assert.equal(headers['X-Hermes-Session-Token'], 'my-token')
})

test('fetchJsonHeaders OMITTS X-Hermes-Session-Token when token is undefined (#63707)', () => {
  // This is the regression guard: the pre-#63707 bug set this header to
  // `undefined` and crashed Node's http.request. The fix omits the key.
  const headers = fetchJsonHeaders(undefined, undefined)
  assert.equal(
    Object.prototype.hasOwnProperty.call(headers, 'X-Hermes-Session-Token'),
    false,
    'X-Hermes-Session-Token must NOT appear in headers when token is undefined',
  )
})

test('fetchJsonHeaders OMITTS X-Hermes-Session-Token when token is null', () => {
  const headers = fetchJsonHeaders(null, undefined)
  assert.equal(
    Object.prototype.hasOwnProperty.call(headers, 'X-Hermes-Session-Token'),
    false,
  )
})

test('fetchJsonHeaders OMITTS X-Hermes-Session-Token when token is empty string', () => {
  const headers = fetchJsonHeaders('', undefined)
  assert.equal(
    Object.prototype.hasOwnProperty.call(headers, 'X-Hermes-Session-Token'),
    false,
    'empty-string token should be treated like nullish (no header)',
  )
})

test('fetchJsonHeaders sets Content-Length when body is present', () => {
  const body = Buffer.from('{"a":1}')
  const headers = fetchJsonHeaders('my-token', body)
  assert.equal(headers['Content-Length'], String(body.length))
})

test('fetchJsonHeaders omits Content-Length when no body', () => {
  const headers = fetchJsonHeaders('my-token', undefined)
  assert.equal(
    Object.prototype.hasOwnProperty.call(headers, 'Content-Length'),
    false,
  )
})

test('fetchJsonHeaders does not include any token-related key when token is nullish', () => {
  // Belt-and-suspenders: ensure the password-auth path produces a clean
  // header set that Node's http.request will accept.
  const headers = fetchJsonHeaders(undefined, undefined)
  const keys = Object.keys(headers)
  assert.deepEqual(keys.sort(), ['Content-Type'].sort())
})

test('fetchJsonHeaders produces a header object http.request accepts (no undefined values)', () => {
  // The pre-#63707 bug surfaced here: Node's http.request throws
  // ERR_INVALID_ARG_TYPE if any header value is undefined. Our fix
  // omits nullish keys entirely, so the produced object is safe to pass
  // straight into http.request without conditional checks at the call site.
  for (const tokenValue of [undefined, null, '']) {
    const headers = fetchJsonHeaders(tokenValue, undefined)
    for (const [key, value] of Object.entries(headers)) {
      assert.notStrictEqual(value, undefined, `header ${key} is undefined for token=${tokenValue}`)
    }
  }
})
