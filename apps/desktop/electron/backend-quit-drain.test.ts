import assert from 'node:assert/strict'

import { test } from 'vitest'

import { collectBackendDrainTargets } from './backend-quit-drain'
import type { BackendChildLike } from './backend-quit-drain'

const alive = (extra: Partial<BackendChildLike> = {}): BackendChildLike => ({
  exitCode: null,
  signalCode: null,
  ...extra,
})

test('captures the primary backend as a drain target when alive', () => {
  const primary = alive()
  const targets = collectBackendDrainTargets(primary, [])
  assert.deepEqual(targets, [primary])
})

test('captures every live pooled backend even though the pool is about to be cleared', () => {
  // Regression for the orphan-on-quit bug: stopAllPoolBackends() deletes the
  // backendPool map entries before SIGTERM, so a caller that reads the pool
  // AFTER clearing gets nothing to wait on. The helper must take its pool
  // snapshot up-front and keep those handles.
  const poolA = alive()
  const poolB = alive()

  const targets = collectBackendDrainTargets(undefined, [
    { process: poolA },
    { process: poolB },
  ])

  assert.deepEqual(targets, [poolA, poolB])
})

test('dedupes a primary that also appears in the pool snapshot', () => {
  const shared = alive()

  const targets = collectBackendDrainTargets(shared, [
    { process: shared },
    { process: alive() },
  ])

  assert.strictEqual(targets.length, 2)
})

test('skips already-exited children (no point waiting on them)', () => {
  const dead = alive({ exitCode: 0, signalCode: null })
  const liveChild = alive()
  const targets = collectBackendDrainTargets(dead, [{ process: liveChild }])
  assert.deepEqual(targets, [liveChild])
})

test('returns an empty list when there is nothing to drain', () => {
  assert.deepEqual(collectBackendDrainTargets(undefined, []), [])
  assert.deepEqual(collectBackendDrainTargets(null, [{ process: null }]), [])
})
