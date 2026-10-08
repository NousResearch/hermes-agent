import assert from 'node:assert/strict'
import test from 'node:test'
import { requestedProfileScope, needsProfileApproval } from './profile-access'
import { claimPersistentProfile } from './owned-browser'

test('private access cannot authorize account mode or a different existing window', () => {
  const grants = new Set<string>(['background'])
  const athena = requestedProfileScope('browser_prepare', { browser: 'edge', profile: { mode: 'athena_profile' } })!
  const existing = requestedProfileScope('browser_prepare', { pid: 10, window_id: 20, profile: { mode: 'existing_profile' } })!
  assert(needsProfileApproval(grants, athena))
  assert(needsProfileApproval(grants, existing))
  grants.add(athena)
  assert(!needsProfileApproval(grants, requestedProfileScope('get_browser_state', {}, athena)))
  assert(needsProfileApproval(grants, existing))
  grants.add(existing)
  assert(needsProfileApproval(grants, requestedProfileScope('get_browser_state', {}, 'existing_profile:10:21')))
  assert(needsProfileApproval(new Set(), athena))
  grants.clear()
  assert(needsProfileApproval(grants, existing))
})

test('a persistent Athena profile cannot be shared across simultaneous conversations', () => {
  const first = claimPersistentProfile('/temporary-test-fixture/athena-profile')
  try { assert.throws(() => claimPersistentProfile('/temporary-test-fixture/athena-profile'), /already in use/) } finally { first() }
  const next = claimPersistentProfile('/temporary-test-fixture/athena-profile')
  next()
})
