/**
 * Tests for electron/pool-eviction.ts — LRU cap accounting for the desktop
 * backend pool. The cap exists to bound SPAWNED local backends (real child
 * processes); process-less descriptor entries (remote/cloud registry sources,
 * per-profile remote overrides) must not count against it, or a roster
 * refresh across N registered remote connections evicts a real local backend
 * that was merely idle past the keepalive window.
 */

import assert from 'node:assert/strict'

import { test } from 'vitest'

import { selectPoolEvictions } from './pool-eviction'

const NOW = 1_000_000
const FRESH_MS = 90_000

/** A spawned local backend entry (has a child process). */
const spawned = (idleMs: number) => ({ process: { pid: 123 }, lastActiveAt: NOW - idleMs })

/** A process-less remote/cloud descriptor entry. */
const descriptor = (idleMs: number) => ({ process: null, lastActiveAt: NOW - idleMs })

test('process-less descriptors do not count toward the cap', () => {
  // 1 real spawned backend idle beyond the keepalive window + 3 remote
  // descriptors: total size (4) exceeds keep (2), but only ONE entry holds a
  // process, so nothing may be evicted. This is the roster-refresh regression:
  // the old size-based accounting evicted the real local backend here.
  const entries: [string, ReturnType<typeof spawned>][] = [
    ['default', spawned(120_000)],
    ['conn:homelab::default', descriptor(0)],
    ['conn:office::default', descriptor(0)],
    ['conn:cloud-a::default', descriptor(0)]
  ]

  assert.deepEqual(selectPoolEvictions(entries, 2, NOW, FRESH_MS), [])
})

test('spawned backends over the cap are still LRU-evicted', () => {
  const entries: [string, ReturnType<typeof spawned>][] = [
    ['a', spawned(500_000)],
    ['b', spawned(300_000)],
    ['c', spawned(100_000)],
    // Descriptors interleaved: must neither inflate the count nor be evicted.
    ['conn:x::a', descriptor(999_000)]
  ]

  // keep=2 → one spawned backend over; evict the least-recently-used ('a').
  assert.deepEqual(selectPoolEvictions(entries, 2, NOW, FRESH_MS), ['a'])
})

test('fresh spawned backends are spared even over the cap', () => {
  const entries: [string, ReturnType<typeof spawned>][] = [
    ['a', spawned(1_000)],
    ['b', spawned(2_000)],
    ['c', spawned(3_000)]
  ]

  // All within the keepalive window → the pool may exceed the soft cap.
  assert.deepEqual(selectPoolEvictions(entries, 1, NOW, FRESH_MS), [])
})

test('evicts only enough stale spawned backends to reach the cap', () => {
  const entries: [string, ReturnType<typeof spawned>][] = [
    ['a', spawned(500_000)],
    ['b', spawned(400_000)],
    ['c', spawned(300_000)],
    ['d', spawned(1_000)]
  ]

  // 4 spawned, keep 2 → remove 2, oldest first.
  assert.deepEqual(selectPoolEvictions(entries, 2, NOW, FRESH_MS), ['a', 'b'])
})

test('descriptor-only pools never evict', () => {
  const entries: [string, ReturnType<typeof descriptor>][] = [
    ['conn:a::p', descriptor(999_000)],
    ['conn:b::p', descriptor(999_000)],
    ['conn:c::p', descriptor(999_000)],
    ['conn:d::p', descriptor(999_000)]
  ]

  assert.deepEqual(selectPoolEvictions(entries, 2, NOW, FRESH_MS), [])
})
