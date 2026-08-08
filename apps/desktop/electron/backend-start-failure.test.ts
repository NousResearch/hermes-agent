import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  createTokenRejectionRetryGuard,
  isSessionTokenRejectionError,
  MAX_TOKEN_REJECTION_BOOT_RETRIES,
  sessionTokenRejectionHint,
  shouldLatchBackendStartFailure,
  shouldLatchRemoteReauthFailure
} from './backend-start-failure'

// The exact message main.ts constructs when the local boot's WS probe fails
// (see the `Local Hermes backend is HTTP-reachable but the WebSocket
// (/api/ws) rejected the session token:` throw site in startHermes).
function tokenRejectionError(message = 'Local Hermes backend is HTTP-reachable but the WebSocket (/api/ws) rejected the session token: WebSocket connection failed.'): Error {
  return new Error(message)
}

test('latches a LOCAL backend failure so the install-retry loop is broken', () => {
  assert.equal(shouldLatchBackendStartFailure({ attemptedRemote: false }), true)
})

test('never latches a REMOTE failure so recovery stays retryable without a restart', () => {
  // A lapsed OAuth session / mint timeout / host briefly unreachable across a
  // laptop sleep must not wedge the app: the next connect has to re-attempt and
  // re-mint against the refreshed session.
  assert.equal(shouldLatchBackendStartFailure({ attemptedRemote: true }), false)
})

test('the two branches are mutually exclusive (a failure either latches or stays retryable)', () => {
  for (const attemptedRemote of [true, false]) {
    const latched = shouldLatchBackendStartFailure({ attemptedRemote })
    assert.equal(latched, !attemptedRemote)
  }
})

test('latches a CONFIRMED remote reauth failure so the overlay stays clickable', () => {
  // Without this the non-latching remote path re-runs startHermes on every
  // getConnection/api call, re-emits running:true, and the overlay hides
  // itself — the "Sign in" button flickers away before it can be clicked.
  assert.equal(shouldLatchRemoteReauthFailure({ attemptedRemote: true, isReauth: true }), true)
})

test('does not latch a transient remote failure as reauth', () => {
  // A mint timeout or a host unreachable across sleep must still self-heal.
  assert.equal(shouldLatchRemoteReauthFailure({ attemptedRemote: true, isReauth: false }), false)
})

test('never latches a LOCAL failure as reauth (that is backendStartFailure job)', () => {
  assert.equal(shouldLatchRemoteReauthFailure({ attemptedRemote: false, isReauth: true }), false)
  assert.equal(shouldLatchRemoteReauthFailure({ attemptedRemote: false, isReauth: false }), false)
})

test('the two latches never fire for the same failure', () => {
  // They are complementary, not overlapping: local failures latch via
  // backendStartFailure, confirmed remote reauth latches via its own flag.
  for (const attemptedRemote of [true, false]) {
    for (const isReauth of [true, false]) {
      const start = shouldLatchBackendStartFailure({ attemptedRemote })
      const reauth = shouldLatchRemoteReauthFailure({ attemptedRemote, isReauth })
      assert.ok(!(start && reauth), `both latched for remote=${attemptedRemote} reauth=${isReauth}`)
    }
  }
})

test('classifies the WS session-token rejection boot failure by its exact message', () => {
  // The primary-backend throw site in startHermes.
  assert.equal(isSessionTokenRejectionError(tokenRejectionError()), true)
  // The profile-pool variant is the same failure class.
  assert.equal(
    isSessionTokenRejectionError(
      new Error(
        'Hermes backend for profile "work" is HTTP-reachable but the WebSocket (/api/ws) rejected the session token: 401'
      )
    ),
    true
  )
})

test('does not classify unrelated boot failures as session-token rejections', () => {
  // Port timeouts, child exits, connection-test WS probes, remote auth gaps —
  // none of these are the stale-.env class and none may trip the retry bound.
  for (const message of [
    'Timed out waiting for the Hermes backend port announcement',
    'Hermes backend exited before it became ready (null).',
    'fetch failed: connection refused on 127.0.0.1:8123',
    'Reached the gateway over HTTP, but the live WebSocket (/api/ws) connection failed: blocked by proxy',
    'Remote Hermes gateway is selected, but no session token is saved.',
    'Hermes backend start was superseded by a newer connection attempt.'
  ]) {
    assert.equal(isSessionTokenRejectionError(new Error(message)), false, message)
  }

  assert.equal(isSessionTokenRejectionError(null), false)
  assert.equal(isSessionTokenRejectionError('plain string, not an Error'), false)
})

test('token-rejection failure surfaces the actionable stale-.env hint', () => {
  const envPath = 'C:\\Users\\test\\AppData\\Local\\hermes\\.env'
  const hint = sessionTokenRejectionHint(envPath)

  assert.match(hint, /HERMES_DASHBOARD_SESSION_TOKEN/)
  assert.match(hint, /C:\\Users\\test\\AppData\\Local\\hermes\\.env/)
  assert.match(hint, /hermes setup/)

  // The full message main.ts surfaces (throw-site message + hint) carries both
  // the rejection class marker AND the fix, so the failure overlay shows it.
  const surfaced = `${tokenRejectionError().message} ${hint}`
  assert.equal(isSessionTokenRejectionError(new Error(surfaced)), true)
  assert.ok(surfaced.includes('Remove that line'))
})

test('retry guard increments per consecutive token rejection and exhausts after N', () => {
  const guard = createTokenRejectionRetryGuard(MAX_TOKEN_REJECTION_BOOT_RETRIES)
  assert.equal(MAX_TOKEN_REJECTION_BOOT_RETRIES, 3)
  assert.equal(guard.count, 0)
  assert.equal(guard.exhausted, false)

  guard.recordFailure(tokenRejectionError())
  guard.recordFailure(tokenRejectionError())
  assert.equal(guard.count, 2)
  assert.equal(guard.exhausted, false)

  // The Nth consecutive rejection flips the bound: the reset loop must stop.
  guard.recordFailure(tokenRejectionError())
  assert.equal(guard.count, 3)
  assert.equal(guard.exhausted, true)

  // Further rejections keep it exhausted — the bound does not un-trip.
  guard.recordFailure(tokenRejectionError())
  assert.equal(guard.exhausted, true)
})

test('non-token-rejection failures keep the existing retry behavior (no regression)', () => {
  const guard = createTokenRejectionRetryGuard(3)

  // A run of unrelated failures never exhausts the bound.
  guard.recordFailure(new Error('Timed out waiting for the Hermes backend port announcement'))
  guard.recordFailure(new Error('Hermes backend exited before it became ready (null).'))
  assert.equal(guard.count, 0)
  assert.equal(guard.exhausted, false)

  // An unrelated failure between rejections breaks the streak, so the bound
  // only ever applies to CONSECUTIVE token rejections.
  guard.recordFailure(tokenRejectionError())
  guard.recordFailure(tokenRejectionError())
  assert.equal(guard.count, 2)
  guard.recordFailure(new Error('fetch failed: connection refused on 127.0.0.1:8123'))
  assert.equal(guard.count, 0)
  assert.equal(guard.exhausted, false)

  // And a clean boot (reset) clears an exhausted guard for the next episode.
  guard.recordFailure(tokenRejectionError())
  guard.recordFailure(tokenRejectionError())
  guard.recordFailure(tokenRejectionError())
  assert.equal(guard.exhausted, true)
  guard.reset()
  assert.equal(guard.count, 0)
  assert.equal(guard.exhausted, false)
})
