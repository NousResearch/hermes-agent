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
// Mirrors main.ts POOL_KEEPALIVE_FRESH_MS (4 minutes — see #95189).
const FRESH_MS = 4 * 60_000

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

// ── #95189 — Keepalive-fresh window must tolerate transient missed pings ──
// Symptom: gateway restarts every ~2 minutes on WSL2. Root cause: the renderer
// pings every 60s; the LRU cap declared a backend "stale" if `lastActiveAt`
// was > 90s ago — only 1.5× the ping interval. WSL2 IPC roundtrips (renderer
// → 9p → Electron main → ipcMain.handle) commonly stall several seconds; one
// delayed or missed ping pushed a live backend past the threshold and the
// cap-driven eviction killed the active profile's backend mid-session,
// forcing a restart loop that re-minted runtime ids and re-allocated pooled
// gateway secondaries ~700×/day (#95189, related #87906/#84716/#88054).
//
// These tests pin the new tolerance: the keepalive-fresh window is wide
// enough to absorb ≥1 missed ping (and the IPC stall headroom around it)
// without evicting an active backend. Truly stale backends (multiple lapses,
// minutes idle) are still evicted as before.

test('#95189: one missed keepalive ping must NOT make the most-recently-touched backend evictable', () => {
  // Renderer's keepalive cadence is 60s. With the old freshMs=90s window,
  // a backend last touched 95s ago — i.e. exactly ONE missed/delayed ping —
  // was eligible for LRU eviction even though it had been actively
  // touched the moment before and would be touched again imminently.
  //
  // Build a pool where the only entry over the cap is the active one (95s
  // idle). With the old 90s window, it was evicted; with the widened window
  // the cap must instead be honored by leaving the pool over-cap for one
  // extra cycle — killing an active backend is far worse than briefly
  // exceeding the soft cap.
  const entries: [string, ReturnType<typeof spawned>][] = [
    ['active', spawned(95_000)], // 1 missed ping on a 60s cadence
    ['fresher', spawned(2_000)] // touched recently — must NOT be evicted
  ]

  // keep=1 → pool over cap by one. Active backend must be spared; the cap
  // may be exceeded rather than kill a live backend (the long-standing
  // "spare fresh backends" rule from #94381 / earlier pool-eviction tests).
  assert.deepEqual(selectPoolEvictions(entries, 1, NOW, FRESH_MS), [])
})

test('#95189: two missed keepalive pings (2-min IPC stall) must NOT evict an active backend', () => {
  // The reported symptom: gateways exited ~80–90s after start with a clean
  // disconnect (no stderr), recurring every ~2 min. Reproduce the boundary:
  // 125s of silence = just over two missed pings at 60s. A backend in this
  // state is still actively serving — the renderer is mid-reconnect, not
  // gone — so eviction here was the trigger for the restart loop.
  //
  // The other entries are all FRESH (well within the keepalive window), so
  // no eviction is correct even before any cap considerations.
  const entries: [string, ReturnType<typeof spawned>][] = [
    ['active', spawned(125_000)],
    ['fresher', spawned(2_000)],
    ['fresher-2', spawned(5_000)]
  ]

  assert.deepEqual(selectPoolEvictions(entries, 1, NOW, FRESH_MS), [])
})

test('#95189: a backend genuinely idle for minutes IS evicted (#95189 long-window scenario)', () => {
  // Sanity: the widening does NOT make the pool unbounded. 10 minutes of
  // silence (the documented POOL_IDLE_MS) is still fair game.
  const entries: [string, ReturnType<typeof spawned>][] = [
    ['idle', spawned(10 * 60_000)],
    ['fresh', spawned(5_000)]
  ]

  // keep=1, idle is over the cap AND past the fresh window → evicted.
  assert.deepEqual(selectPoolEvictions(entries, 1, NOW, FRESH_MS), ['idle'])
})
