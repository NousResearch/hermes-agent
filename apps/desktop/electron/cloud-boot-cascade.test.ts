/**
 * Regression tests for electron/cloud-boot-cascade.ts.
 *
 * A Hermes Cloud agent whose session cookie had expired failed at boot with
 * "Remote Hermes gateway uses OAuth, but you are not signed in" and latched,
 * although the portal session needed to silently re-mint it was live. These
 * pin the exact conditions under which boot is allowed to self-heal.
 *
 * Run via the vitest `electron` project (electron/**\/*.test.ts).
 */

import assert from 'node:assert/strict'

import { test } from 'vitest'

import { makeReauthRequiredError, makeUnsignedOauthError } from './backend-health'
import { shouldAttemptCloudBootCascade } from './cloud-boot-cascade'

const cloudOauth = { remoteKind: 'cloud', authMode: 'oauth' }

test('cloud oauth agent + unsigned reauth error => run the cascade', () => {
  assert.equal(shouldAttemptCloudBootCascade(cloudOauth, makeUnsignedOauthError()), true)
})

test('cloud oauth agent + expired-session reauth error => run the cascade', () => {
  assert.equal(shouldAttemptCloudBootCascade(cloudOauth, makeReauthRequiredError('ticket 401')), true)
})

test('a plain url remote never triggers the cascade, even with the same error', () => {
  assert.equal(shouldAttemptCloudBootCascade({ remoteKind: 'url', authMode: 'oauth' }, makeUnsignedOauthError()), false)
})

test('a token-auth cloud connection has no cookie to mint, so no cascade', () => {
  assert.equal(shouldAttemptCloudBootCascade({ remoteKind: 'cloud', authMode: 'token' }, makeUnsignedOauthError()), false)
})

test('transport and server errors keep their existing handling', () => {
  assert.equal(shouldAttemptCloudBootCascade(cloudOauth, new Error('fetch failed')), false)

  const serverSide = new Error('503: upstream unavailable') as any

  serverSide.statusCode = 503
  assert.equal(shouldAttemptCloudBootCascade(cloudOauth, serverSide), false)
})

test('a bare needsOauthLogin hint without the reauth latch does not qualify', () => {
  // The IPC-shaped hint only drives Sign in copy; only the terminal latch
  // means the ticket mint actually rejected the session.
  const hint = new Error('sign in') as any

  hint.needsOauthLogin = true
  assert.equal(shouldAttemptCloudBootCascade(cloudOauth, hint), false)
})

test('missing or malformed connection objects fail closed', () => {
  assert.equal(shouldAttemptCloudBootCascade(null, makeUnsignedOauthError()), false)
  assert.equal(shouldAttemptCloudBootCascade(undefined, makeUnsignedOauthError()), false)
  assert.equal(shouldAttemptCloudBootCascade('cloud' as any, makeUnsignedOauthError()), false)
  assert.equal(shouldAttemptCloudBootCascade({}, makeUnsignedOauthError()), false)
})
