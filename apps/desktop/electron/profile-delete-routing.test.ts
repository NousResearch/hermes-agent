import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  completeProfileAppearanceCleanup,
  decideProfileDeleteAction,
  profileNameFromCreateRequest,
  profileNameFromDeleteRequest,
  resolveRouteProfile
} from './profile-delete-routing'

// ---------------------------------------------------------------------------
// profileNameFromDeleteRequest
// ---------------------------------------------------------------------------

test('profileNameFromDeleteRequest parses a DELETE /api/profiles/<name> path', () => {
  assert.equal(profileNameFromDeleteRequest({ method: 'DELETE', path: '/api/profiles/worker' }), 'worker')
})

test('profileNameFromDeleteRequest lowercases the profile name', () => {
  assert.equal(profileNameFromDeleteRequest({ method: 'DELETE', path: '/api/profiles/Worker' }), 'worker')
})

test('profileNameFromDeleteRequest returns null for non-DELETE methods', () => {
  assert.equal(profileNameFromDeleteRequest({ method: 'GET', path: '/api/profiles/worker' }), null)
})

test('profileNameFromDeleteRequest returns null when the path does not match', () => {
  assert.equal(profileNameFromDeleteRequest({ method: 'DELETE', path: '/api/sessions' }), null)
})

test('profileNameFromDeleteRequest returns null for an empty/whitespace name', () => {
  assert.equal(profileNameFromDeleteRequest({ method: 'DELETE', path: '/api/profiles/%20' }), null)
})

test('profileNameFromDeleteRequest returns null for an undecodable path segment', () => {
  assert.equal(profileNameFromDeleteRequest({ method: 'DELETE', path: '/api/profiles/%E0%A4%A' }), null)
})

test('profileNameFromCreateRequest parses and normalizes POST /api/profiles', () => {
  assert.equal(
    profileNameFromCreateRequest({ body: { name: ' Worker ' }, method: 'POST', path: '/api/profiles' }),
    'worker'
  )
  assert.equal(profileNameFromCreateRequest({ body: { name: 'worker' }, method: 'GET', path: '/api/profiles' }), null)
  assert.equal(profileNameFromCreateRequest({ body: {}, method: 'POST', path: '/api/profiles' }), null)
  assert.equal(profileNameFromCreateRequest({ body: { name: 'default' }, method: 'POST', path: '/api/profiles' }), null)
  assert.equal(
    profileNameFromCreateRequest({ body: { name: 'worker' }, method: 'POST', path: '/api/profiles/worker' }),
    null
  )
})

// ---------------------------------------------------------------------------
// decideProfileDeleteAction
// ---------------------------------------------------------------------------

const deps = {
  isDefaultProfile: p => p === 'default',
  isValidProfileName: p => /^[a-z0-9][a-z0-9_-]{0,63}$/.test(p),
  primaryProfileKey: () => 'primary-profile'
}

test('decideProfileDeleteAction is a noop for the default profile', () => {
  assert.deepEqual(decideProfileDeleteAction('default', deps), { action: 'noop', profile: null })
})

test('decideProfileDeleteAction is a noop for null (no profile parsed)', () => {
  assert.deepEqual(decideProfileDeleteAction(null, deps), { action: 'noop', profile: null })
})

test('decideProfileDeleteAction is a noop for an invalid profile name', () => {
  assert.deepEqual(decideProfileDeleteAction('Not Valid!', deps), { action: 'noop', profile: null })
})

test('decideProfileDeleteAction tears down the primary backend for the primary profile', () => {
  assert.deepEqual(decideProfileDeleteAction('primary-profile', deps), {
    action: 'teardown-primary',
    profile: 'primary-profile'
  })
})

test('decideProfileDeleteAction tears down the pool backend for any other valid profile', () => {
  assert.deepEqual(decideProfileDeleteAction('worker', deps), { action: 'teardown-pool', profile: 'worker' })
})

// ---------------------------------------------------------------------------
// resolveRouteProfile
// ---------------------------------------------------------------------------

test('resolveRouteProfile routes to the primary backend (null) when a profile was torn down', () => {
  assert.equal(resolveRouteProfile('worker', 'other-profile'), null)
})

test('resolveRouteProfile passes the requested profile through when nothing was torn down', () => {
  assert.equal(resolveRouteProfile(null, 'other-profile'), 'other-profile')
})

test('resolveRouteProfile passes through undefined when nothing was torn down and no profile was requested', () => {
  assert.equal(resolveRouteProfile(null, undefined), undefined)
})

test('completeProfileAppearanceCleanup removes local wallpaper and notifies renderers after success', async () => {
  const removed: string[] = []
  const notified: string[] = []
  const result = { ok: true }

  assert.equal(
    await completeProfileAppearanceCleanup('worker', result, {
      logCleanupFailure: () => assert.fail('cleanup should not fail'),
      notifyProfileReset: profile => notified.push(profile),
      removeWallpaper: async profile => removed.push(profile)
    }),
    result
  )
  assert.deepEqual(removed, ['worker'])
  assert.deepEqual(notified, ['worker'])
})

test('completeProfileAppearanceCleanup preserves success and purges preferences when file cleanup fails', async () => {
  const failures: string[] = []
  const notified: string[] = []
  const result = { ok: true }

  assert.equal(
    await completeProfileAppearanceCleanup('worker', result, {
      logCleanupFailure: profile => failures.push(profile),
      notifyProfileReset: profile => notified.push(profile),
      removeWallpaper: async () => {
        throw new Error('disk unavailable')
      }
    }),
    result
  )
  assert.deepEqual(failures, ['worker'])
  assert.deepEqual(notified, ['worker'])
})

test('completeProfileAppearanceCleanup preserves success when a renderer closes during notification', async () => {
  const failures: string[] = []
  const result = { ok: true }

  assert.equal(
    await completeProfileAppearanceCleanup('worker', result, {
      logCleanupFailure: profile => failures.push(profile),
      notifyProfileReset: () => {
        throw new Error('renderer was destroyed')
      },
      removeWallpaper: async () => undefined
    }),
    result
  )
  assert.deepEqual(failures, ['worker'])
})

test('completeProfileAppearanceCleanup is a no-op for unrelated API requests', async () => {
  const result = { sessions: [] }

  assert.equal(
    await completeProfileAppearanceCleanup(null, result, {
      logCleanupFailure: () => assert.fail('no cleanup expected'),
      notifyProfileReset: () => assert.fail('no notification expected'),
      removeWallpaper: async () => assert.fail('no removal expected')
    }),
    result
  )
})
