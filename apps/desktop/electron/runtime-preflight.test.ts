import assert from 'node:assert/strict'

import { test } from 'vitest'

import { canImportHermesCli } from './backend-probes'
import { bootstrapRuntimeStatus } from './runtime-preflight'

test('valid bootstrap marker plus a real non-Hermes runtime selects repair', () => {
  const runtimeUsable = () => canImportHermesCli(process.execPath)

  assert.equal(runtimeUsable(), false)
  assert.equal(
    bootstrapRuntimeStatus({
      marker: { pinnedCommit: 'abcdef0', schemaVersion: 1 },
      markerSchemaVersion: 1,
      runtimeUsable
    }),
    'repair'
  )
})

test('valid bootstrap marker plus a usable runtime selects the active install', () => {
  assert.equal(
    bootstrapRuntimeStatus({
      marker: { pinnedCommit: 'abcdef0', schemaVersion: 1 },
      markerSchemaVersion: 1,
      runtimeUsable: () => true
    }),
    'ready'
  )
})

test('missing or stale markers leave resolution to external candidates', () => {
  for (const marker of [null, {}, { pinnedCommit: 'abcdef0', schemaVersion: 2 }, { pinnedCommit: '', schemaVersion: 1 }]) {
    assert.equal(
      bootstrapRuntimeStatus({ marker, markerSchemaVersion: 1, runtimeUsable: () => { throw new Error('must not probe') } }),
      'absent'
    )
  }
})
