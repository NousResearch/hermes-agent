/**
 * Tests for collectProcessTreePids in electron/windows-update-runtime.ts —
 * the taskkill-/T-replacement that enumerates a kill tree from a CIM
 * inventory with the caller's own pid hard-excluded.
 *
 * What this locks:
 *   1. Plain descendant enumeration (children + grandchildren, no strays).
 *   2. The PID-reuse self-kill guard: when a process in the tree has recycled
 *      the desktop's dead launcher-parent PID, the desktop (excluded) and its
 *      Electron children are NOT collected — the exact "all Hermes processes
 *      die within 1s of Update now" incident.
 *   3. Robustness: cycles terminate, excluded root yields nothing.
 */

import assert from 'node:assert/strict'
import { test } from 'vitest'

import { collectProcessTreePids } from './windows-update-runtime'

function proc(pid: number, parentPid: number, creationTimeMs = pid): Record<string, number> {
  return { ProcessId: pid, ParentProcessId: parentPid, CreationTimeMs: creationTimeMs }
}

test('collects the root and all transitive descendants', () => {
  const inventory = [proc(100, 1), proc(200, 100), proc(201, 100), proc(300, 200), proc(999, 1)]

  assert.deepEqual(collectProcessTreePids(inventory, 100, { expectedRootCreationTimeMs: 100 }).sort(), [100, 200, 201, 300])
})

test('recycled launcher-parent pid cannot pull the excluded desktop into the kill tree', () => {
  // Desktop 5000 was started by launcher 4000 (dead). Backend 100 spawned a
  // short-lived child that recycled pid 4000 — raw ppid links now claim the
  // desktop descends from the backend. taskkill /T would kill 5000 + its
  // Electron children; the exclude guard must keep them all out.
  const inventory = [
    proc(100, 5000, 1_000), // backend (child of desktop)
    proc(4000, 100, 2_000), // backend child that RECYCLED the dead launcher pid
    proc(5000, 4000, 500), // desktop is older: current 4000 cannot be its real parent
    proc(5001, 5000, 600), // Electron renderer
    proc(5002, 5000, 700) // Electron gpu
  ]

  const tree = collectProcessTreePids(inventory, 100, {
    excludePids: [5000],
    expectedRootCreationTimeMs: 1_000
  })

  assert.deepEqual(tree.sort(), [100, 4000])
  assert.ok(!tree.includes(5000))
  assert.ok(!tree.includes(5001))
  assert.ok(!tree.includes(5002))
})

test('parent-link cycles terminate and excluded root returns empty', () => {
  const cycle = [proc(10, 20), proc(20, 10)]

  assert.deepEqual(collectProcessTreePids(cycle, 10, { expectedRootCreationTimeMs: 10 }).sort(), [10, 20])
  assert.deepEqual(
    collectProcessTreePids(cycle, 10, { excludePids: [10], expectedRootCreationTimeMs: 10 }),
    []
  )
  assert.deepEqual(collectProcessTreePids([], 0, { expectedRootCreationTimeMs: 1 }), [])
})

test('missing creation evidence never expands a raw PPID edge or kills a recycled root', () => {
  const missingTimes = [
    { ProcessId: 100, ParentProcessId: 1 },
    { ProcessId: 200, ParentProcessId: 100 }
  ]

  assert.deepEqual(collectProcessTreePids(missingTimes, 100, { expectedRootCreationTimeMs: 1_000 }), [])
  assert.deepEqual(
    collectProcessTreePids([proc(200, 100, 2_000)], 100, { expectedRootCreationTimeMs: 1_000 }),
    []
  )
})

test('a recycled root PID is rejected unless its creation time matches the tracked child', () => {
  const recycled = [proc(100, 1, 9_000), proc(200, 100, 9_100)]

  assert.deepEqual(collectProcessTreePids(recycled, 100, { expectedRootCreationTimeMs: 1_000 }), [])
  assert.deepEqual(collectProcessTreePids(recycled, 100, { expectedRootCreationTimeMs: 9_000 }), [100, 200])
})
