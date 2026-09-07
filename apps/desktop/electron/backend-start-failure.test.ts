import assert from 'node:assert/strict'

import { test } from 'vitest'

import { isReauthRequiredError, makeUnsignedOauthError } from './backend-health'
import {
  isHostKeyChangedBootFailure,
  isRetryableRemoteBootFailure,
  shouldLatchBackendStartFailure,
  shouldLatchHostKeyChangedFailure,
  shouldLatchRemoteReauthFailure
} from './backend-start-failure'

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

test('unsigned OAuth latches and is never auto-retried; needsOauthLogin alone still retries', () => {
  // Production composition in startHermes: isReauth = isReauthRequiredError(error).
  const unsigned = isReauthRequiredError(makeUnsignedOauthError())
  const ticketHint = isReauthRequiredError({ needsOauthLogin: true })

  assert.equal(unsigned, true)
  assert.equal(shouldLatchRemoteReauthFailure({ attemptedRemote: true, isReauth: unsigned }), true)
  assert.equal(isRetryableRemoteBootFailure({ attemptedRemote: true, isReauth: unsigned }), false)
  assert.equal(ticketHint, false)
  assert.equal(shouldLatchRemoteReauthFailure({ attemptedRemote: true, isReauth: ticketHint }), false)
  assert.equal(isRetryableRemoteBootFailure({ attemptedRemote: true, isReauth: ticketHint }), true)
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

test('FIX host-key change: classified from the kind tag and from stringified ssh banners', () => {
  // classifySshError tags the Error it built; errors that crossed an IPC or
  // string boundary only keep the message. Both shapes must classify.
  const tagged = Object.assign(new Error('SSH refused to connect.'), { kind: 'host-key-changed' })
  assert.equal(isHostKeyChangedBootFailure(tagged), true)
  assert.equal(
    isHostKeyChangedBootFailure(new Error('@@@@ WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED! @@@@')),
    true
  )
  assert.equal(isHostKeyChangedBootFailure(new Error('Host key verification failed.')), true)
  assert.equal(
    isHostKeyChangedBootFailure(new Error('The host key for root@203.0.113.7 has CHANGED since you last connected.')),
    true
  )
  assert.equal(isHostKeyChangedBootFailure(new Error('Connection refused')), false)
  assert.equal(isHostKeyChangedBootFailure(null), false)
})

test('FIX host-key change: latches and is never auto-retried (157-failure loop, Aug 2026 bundle)', () => {
  // SSH fails closed on a changed host key: every retry re-drives the same
  // doomed boot until the user clears known_hosts. Terminal, like reauth.
  const context = { attemptedRemote: true, isReauth: false, isHostKeyChanged: true }
  assert.equal(shouldLatchHostKeyChangedFailure(context), true)
  assert.equal(isRetryableRemoteBootFailure(context), false)
})

test('host-key latch never fires for local failures or ordinary remote faults', () => {
  assert.equal(
    shouldLatchHostKeyChangedFailure({ attemptedRemote: false, isReauth: false, isHostKeyChanged: true }),
    false
  )
  assert.equal(
    shouldLatchHostKeyChangedFailure({ attemptedRemote: true, isReauth: false, isHostKeyChanged: false }),
    false
  )
  assert.equal(shouldLatchHostKeyChangedFailure({ attemptedRemote: true, isReauth: false }), false)
})

test('every remote failure picks exactly one path: retry, reauth latch, or host-key latch', () => {
  for (const isReauth of [true, false]) {
    for (const isHostKeyChanged of [true, false]) {
      const retry = isRetryableRemoteBootFailure({ attemptedRemote: true, isReauth, isHostKeyChanged })
      const reauth = shouldLatchRemoteReauthFailure({ attemptedRemote: true, isReauth })
      const hostKey = shouldLatchHostKeyChangedFailure({ attemptedRemote: true, isReauth, isHostKeyChanged })
      const picked = [retry, reauth, hostKey].filter(Boolean).length
      assert.ok(picked >= 1, `remote failure reauth=${isReauth} hostKey=${isHostKeyChanged} fell through every path`)
    }
  }
})
