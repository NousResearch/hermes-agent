/**
 * Regression tests for electron/native-auth-decisions.ts — the three pure
 * decision seams behind the RFC 8252 native-app auth flow, each of which was a
 * real runtime bug that the mocked flow tests could not catch.
 *
 * Run via the vitest `electron` project (electron/**\/*.test.ts).
 */

import assert from 'node:assert/strict'

import { test } from 'vitest'

import { oauthSessionIsLive, resolveJsonBody, resolveOauthRestAuth } from './native-auth-decisions'

// --- 1. body encoding (guards the double-JSON.stringify 422) ---

test('resolveJsonBody returns the object unchanged (no pre-stringify)', () => {
  const body = { code: 'abc', code_verifier: 'xyz' }
  const out = resolveJsonBody(body)

  // Must be the SAME object reference / shape — NOT a JSON string. Pre-
  // stringifying here is what produced the gateway 422 "Input should be a
  // valid dictionary" at /auth/native/token.
  assert.equal(typeof out, 'object')
  assert.deepEqual(out, body)
})

test('resolveJsonBody does not stringify — a string stays a string, an object stays an object', () => {
  assert.equal(typeof resolveJsonBody({ a: 1 }), 'object')
  // If a caller ever passes an already-encoded string (the bug), we return it
  // as-is rather than re-wrapping — the contract is "fetchJson owns encoding".
  assert.equal(typeof resolveJsonBody('{"a":1}'), 'string')
})

// --- 2. oauth liveness (guards the needsOauthLogin loop) ---

test('oauthSessionIsLive is true when a native bearer token exists, even with no cookie', () => {
  // The exact bug: native login stores a bearer, sets no cookie. Gating on the
  // cookie alone looped the UI into "not signed in".
  assert.equal(oauthSessionIsLive(true, false), true)
})

test('oauthSessionIsLive is true when a live cookie exists with no native token', () => {
  assert.equal(oauthSessionIsLive(false, true), true)
})

test('oauthSessionIsLive is true when both are present', () => {
  assert.equal(oauthSessionIsLive(true, true), true)
})

test('oauthSessionIsLive is false only when neither is present', () => {
  assert.equal(oauthSessionIsLive(false, false), false)
})

// --- 3. REST auth selection (guards the 401 no_cookie) ---

test('resolveOauthRestAuth prefers the native bearer when a token is present', () => {
  const auth = resolveOauthRestAuth('bearer-token-123')

  assert.deepEqual(auth, { kind: 'bearer', token: 'bearer-token-123' })
})

test('resolveOauthRestAuth falls back to cookie when there is no native token', () => {
  assert.deepEqual(resolveOauthRestAuth(null), { kind: 'cookie' })
  assert.deepEqual(resolveOauthRestAuth(undefined), { kind: 'cookie' })
  // Empty string is not a usable bearer — must fall back, not send "Bearer ".
  assert.deepEqual(resolveOauthRestAuth(''), { kind: 'cookie' })
})
