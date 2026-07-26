/**
 * Regression tests for electron/native-auth-decisions.ts — the pure decision
 * seams behind the RFC 8252 native-app auth flow, each of which was a real
 * runtime bug that the mocked flow tests could not catch.
 *
 * Run via the vitest `electron` project (electron/**\/*.test.ts).
 */

import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  oauthGuardMayHardFail,
  oauthSessionIsLive,
  resolveJsonBody,
  resolveOauthRestAuth,
  resolveReadinessProbeAuth
} from './native-auth-decisions'

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

// --- 4. readiness-probe auth (guards the credential-free 401 boot loop) ---

test('resolveReadinessProbeAuth reuses the oauth bearer-vs-cookie choice', () => {
  assert.deepEqual(resolveReadinessProbeAuth('oauth', 'native-at'), { kind: 'bearer', token: 'native-at' })
  assert.deepEqual(resolveReadinessProbeAuth('oauth', null), { kind: 'cookie' })
  assert.deepEqual(resolveReadinessProbeAuth('oauth', ''), { kind: 'cookie' })
})

test('resolveReadinessProbeAuth sends the session token for a token gateway', () => {
  assert.deepEqual(resolveReadinessProbeAuth('token', null, 'session-token'), {
    kind: 'token',
    token: 'session-token'
  })
  assert.deepEqual(resolveReadinessProbeAuth('token', null, null), { kind: 'token', token: null })
})

test('resolveReadinessProbeAuth stays public for local and unknown modes', () => {
  // A loopback backend has no gate; sending credentials it never issued is
  // meaningless, and an unknown mode must not invent a credential.
  assert.deepEqual(resolveReadinessProbeAuth('local', 'native-at', 'session-token'), { kind: 'public' })
  assert.deepEqual(resolveReadinessProbeAuth(undefined, 'native-at', 'session-token'), { kind: 'public' })
  assert.deepEqual(resolveReadinessProbeAuth('something-new', null, null), { kind: 'public' })
})

// --- 5. oauth guard vs password gateways (guards the false "not signed in") ---

test('oauthGuardMayHardFail is false only when EVERY provider is password-based', () => {
  assert.equal(oauthGuardMayHardFail([{ name: 'basic', supportsPassword: true }]), false)
  assert.equal(
    oauthGuardMayHardFail([
      { name: 'basic', supportsPassword: true },
      { name: 'ldap', supportsPassword: true }
    ]),
    false
  )
})

test('oauthGuardMayHardFail keeps the strict guard for oauth and mixed deployments', () => {
  assert.equal(oauthGuardMayHardFail([{ name: 'nous', supportsPassword: false }]), true)
  assert.equal(
    oauthGuardMayHardFail([
      { name: 'nous', supportsPassword: false },
      { name: 'basic', supportsPassword: true }
    ]),
    true
  )
})

test('oauthGuardMayHardFail keeps the strict guard when the list is unusable', () => {
  // Backends predating /api/auth/providers, or an unreachable probe, must not
  // silently weaken the guard.
  assert.equal(oauthGuardMayHardFail([]), true)
  assert.equal(oauthGuardMayHardFail(null), true)
  assert.equal(oauthGuardMayHardFail(undefined), true)
  assert.equal(oauthGuardMayHardFail('nonsense' as any), true)
  assert.equal(oauthGuardMayHardFail([{ supportsPassword: true }]), true)
})
