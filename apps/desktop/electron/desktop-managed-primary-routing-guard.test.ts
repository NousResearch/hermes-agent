import assert from 'node:assert/strict'

import { test } from 'vitest'

import { createManagedPrimaryRoutingGuard } from './desktop-managed-primary-routing-guard'

test('primary route mutation guard sees live managed update and durable recovery identities', () => {
  const updates = new Map<string, Promise<unknown>>()
  const recoveries = new Map<string, Promise<void>>()
  const restoreOwners = new Map<string, unknown>()
  let durable = [{ connectionId: 'journal' }]

  const assertCanMutate = createManagedPrimaryRoutingGuard({
    managedConnectionUpdates: updates,
    managedConnectionRecoveries: recoveries,
    managedPrimaryRestoreOwners: restoreOwners,
    readManagedSshRecoveryRecords: () => durable
  })

  updates.set('active', Promise.resolve())
  assert.throws(assertCanMutate, (error: any) => error.code === 'managed-update-in-progress' && /active.*journal/.test(error.message))
  updates.clear()
  durable = []
  assert.doesNotThrow(assertCanMutate)
})
