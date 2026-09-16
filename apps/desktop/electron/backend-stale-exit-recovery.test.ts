import assert from 'node:assert/strict'
import { test } from 'vitest'

import { claimStaleBackendExitRecovery } from './backend-stale-exit-recovery'

test('current owner exit is eligible for recovery', () => {
  assert.equal(claimStaleBackendExitRecovery({ hasCurrentProcess: false, hasPendingStart: false, intentionalTeardown: false, recoveryClaimed: false }), true)
})

test('stale exit with a new owner is ignored', () => {
  assert.equal(claimStaleBackendExitRecovery({ hasCurrentProcess: true, hasPendingStart: false, intentionalTeardown: false, recoveryClaimed: false }), false)
})

test('stale exit with an empty state recovers once', () => {
  const state = { hasCurrentProcess: false, hasPendingStart: false, intentionalTeardown: false, recoveryClaimed: false }
  assert.equal(claimStaleBackendExitRecovery(state), true)
  assert.equal(claimStaleBackendExitRecovery(state), false)
})

test('stale exit during intentional teardown does not recover', () => {
  assert.equal(claimStaleBackendExitRecovery({ hasCurrentProcess: false, hasPendingStart: false, intentionalTeardown: true, recoveryClaimed: false }), false)
})

test('double stale exits coalesce to one recovery', () => {
  const state = { hasCurrentProcess: false, hasPendingStart: false, intentionalTeardown: false, recoveryClaimed: false }
  assert.equal([claimStaleBackendExitRecovery(state), claimStaleBackendExitRecovery(state)].filter(Boolean).length, 1)
})

test('current exit while a replacement start is pending does not recover again', () => {
  assert.equal(claimStaleBackendExitRecovery({ hasCurrentProcess: false, hasPendingStart: true, intentionalTeardown: false, recoveryClaimed: false }), false)
})
