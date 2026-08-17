import assert from 'node:assert/strict'

import { test } from 'vitest'

import { isRetryableRemoteBootFailure, shouldLatchBackendStartFailure, shouldLatchRemoteReauthFailure } from './backend-start-failure'

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

test('FIX #82679: a transient remote failure is retryable so a dropped SSH/HTTP connection self-heals', () => {
  // The dropped-registered-connection class: "Could not verify the existing
  // SSH backend", ERR_CONNECTION_RESET on an HTTP remote, mint timeouts. All
  // surface as non-reauth remote boot failures and must enter the bounded
  // renderer retry loop instead of parking on "Desktop boot failed".
  assert.equal(isRetryableRemoteBootFailure({ attemptedRemote: true, isReauth: false }), true)
})

test('a CONFIRMED reauth rejection is never auto-retried (missing capability, not transient failure)', () => {
  assert.equal(isRetryableRemoteBootFailure({ attemptedRemote: true, isReauth: true }), false)
})

test('local failures are never auto-retried by the remote self-heal loop', () => {
  assert.equal(isRetryableRemoteBootFailure({ attemptedRemote: false, isReauth: false }), false)
  assert.equal(isRetryableRemoteBootFailure({ attemptedRemote: false, isReauth: true }), false)
})

test('retryable and reauth-latch are mutually exclusive for remote failures', () => {
  // Every remote failure either self-heals (transient) or latches for sign-in
  // (confirmed reauth) — never both, never neither.
  for (const isReauth of [true, false]) {
    const retry = isRetryableRemoteBootFailure({ attemptedRemote: true, isReauth })
    const latch = shouldLatchRemoteReauthFailure({ attemptedRemote: true, isReauth })
    assert.equal(retry !== latch, true, `remote failure with reauth=${isReauth} must pick exactly one path`)
  }
})
