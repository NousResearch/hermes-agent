// Multiplex-only invariants for the Desktop backend pool: one host backend
// serves every profile, and the local pooled-spawn path is unreachable.
import assert from 'node:assert/strict'

import { test } from 'vitest'

import { resolveProfileBackendRoute } from './connection-config'
import { assertNoSecondLocalBackend, SecondLocalBackendError, sharesHostBackend } from './host-backend-singleton'

const LOCAL = { globalRemote: false, primaryProfile: 'default', profileRemoteOverride: false }

test('two local profiles connecting concurrently produce ZERO additional backends, each bound to its own profile', () => {
  // The routing decision is the whole spawn decision: `ensureBackend` spawns a
  // pooled child if and only if the route says `pool`. Resolve both profiles
  // the way two concurrent renderer dials would.
  const routes = ['worker', 'venture'].map(profile => resolveProfileBackendRoute(profile, LOCAL))

  assert.deepEqual(
    routes.filter(route => route.backend === 'pool'),
    [],
    'a local profile must never resolve to a pooled backend of its own'
  )

  // Both land on the SAME backend and still carry distinct wire identities, so
  // each connection's turns bind to its own home (`session.create {profile}` ->
  // `profile_home`; sessionless RPCs take the explicit `profile` argument).
  assert.deepEqual(
    routes.map(route => [route.backend, route.descriptorProfile, route.scopePath]),
    [
      ['primary', 'worker', true],
      ['primary', 'venture', true]
    ]
  )
})

test('the local pool spawn path is unreachable, and the escape hatches still reach it', () => {
  assert.throws(() => assertNoSecondLocalBackend('worker', { isolated: false }), SecondLocalBackendError)

  // HERMES_DESKTOP_ISOLATED_BACKEND=1: a private backend for this app.
  assert.equal(sharesHostBackend({ isolated: true }), false)
  assertNoSecondLocalBackend('worker', { isolated: true })
  assert.equal(resolveProfileBackendRoute('worker', { ...LOCAL, isolatedBackend: true }).backend, 'pool')

  // A remote/SSH backend is a DIFFERENT host: out of scope for the singleton,
  // and its pooled descriptor never meant a local child anyway.
  assert.equal(sharesHostBackend({ profileRemoteOverride: true }), false)
  assert.equal(sharesHostBackend({ primaryRemoteActive: true }), false)
  assertNoSecondLocalBackend('worker', { profileRemoteOverride: true })
  assert.equal(resolveProfileBackendRoute('worker', { ...LOCAL, profileRemoteOverride: true }).backend, 'pool')
})
