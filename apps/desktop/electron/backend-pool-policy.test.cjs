const test = require('node:test')
const assert = require('node:assert/strict')

const {
  createPoolBackendEntry,
  isPoolBackendEvictable,
  isPoolBackendReapable,
  poolBackendHasRendererLease,
  releasePoolBackendEntry,
  retainPoolBackendEntry,
  touchPoolBackendEntry
} = require('./backend-pool-policy.cjs')

test('pool backend entries start unleased and recently active', () => {
  const entry = createPoolBackendEntry({ now: 1_000 })

  assert.equal(entry.lastActiveAt, 1_000)
  assert.equal(entry.rendererLeaseCount, 0)
  assert.equal(poolBackendHasRendererLease(entry), false)
})

test('idle reaper may reap an unleased backend after the idle window', () => {
  const entry = createPoolBackendEntry({ now: 1_000 })

  assert.equal(isPoolBackendReapable(entry, { now: 1_000 + 600_000, idleMs: 600_000 }), false)
  assert.equal(isPoolBackendReapable(entry, { now: 1_000 + 600_001, idleMs: 600_000 }), true)
})

test('renderer leases prevent idle reaping even when heartbeats are delayed', () => {
  const entry = createPoolBackendEntry({ now: 1_000 })
  retainPoolBackendEntry(entry, { now: 2_000 })

  assert.equal(poolBackendHasRendererLease(entry), true)
  assert.equal(isPoolBackendReapable(entry, { now: 10_000_000, idleMs: 600_000 }), false)
})

test('lease release restores normal idle reaping and clamps duplicate releases', () => {
  const entry = createPoolBackendEntry({ now: 1_000 })
  retainPoolBackendEntry(entry, { now: 2_000 })

  assert.equal(releasePoolBackendEntry(entry, { now: 3_000 }), 0)
  assert.equal(releasePoolBackendEntry(entry, { now: 4_000 }), 0)
  assert.equal(poolBackendHasRendererLease(entry), false)
  assert.equal(isPoolBackendReapable(entry, { now: 604_001, idleMs: 600_000 }), true)
})

test('LRU eviction also spares renderer-leased backends', () => {
  const entry = createPoolBackendEntry({ now: 1_000 })
  retainPoolBackendEntry(entry, { now: 2_000 })

  assert.equal(isPoolBackendEvictable(entry, { now: 10_000_000, freshMs: 90_000 }), false)

  releasePoolBackendEntry(entry, { now: 3_000 })
  assert.equal(isPoolBackendEvictable(entry, { now: 94_001, freshMs: 90_000 }), true)
})

test('touch updates last activity without creating a lease', () => {
  const entry = createPoolBackendEntry({ now: 1_000 })
  touchPoolBackendEntry(entry, { now: 10_000 })

  assert.equal(entry.lastActiveAt, 10_000)
  assert.equal(entry.rendererLeaseCount, 0)
  assert.equal(isPoolBackendReapable(entry, { now: 610_000, idleMs: 600_000 }), false)
})
