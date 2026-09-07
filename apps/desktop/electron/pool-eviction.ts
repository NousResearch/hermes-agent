// LRU cap accounting for the desktop backend pool.
//
// The pool holds two very different kinds of entries under one Map:
//   1. SPAWNED local profile backends — a real child process each (the thing
//      the POOL_MAX_BACKENDS cap exists to bound).
//   2. Process-less connection DESCRIPTORS — remote/cloud registry sources and
//      per-profile remote overrides (`entry.process === null`). These hold no
//      local process; their only cost is a cached descriptor.
//
// Counting both kinds against the cap meant a roster refresh across N
// registered remote connections could push the Map size over the cap and
// LRU-evict a REAL spawned backend that had merely been idle past the
// keepalive window. Cap accounting (and cap-driven eviction) therefore only
// considers entries with a live child process; descriptor entries remain
// subject to the idle reaper, just not to the process cap.

export interface PoolEvictionEntry {
  lastActiveAt?: null | number
  process?: unknown
}

// Hard-cap padding for keepalive-pinned backends. The soft cap (maxBackends)
// spares keepalive-fresh entries unconditionally — when every backend is
// actively kept alive the pool can exceed the soft cap rather than kill a
// running session. Without a second tier this is unbounded: every profile
// whose chat was ever opened keeps a resident serve process (~120 MB) until
// app quit (issue #105239: 62 profiles → 126 processes, ~7.5 GB). The hard
// cap bounds the pinned tier so the pool's memory footprint is always capped
// regardless of how many profile chats have been opened over the app's
// lifetime. Soft = maxBackends, hard = maxBackends + HARD_CAP_EXTRA.
const HARD_CAP_EXTRA = 6

/**
 * Pick which pool keys the LRU cap should evict so that at most `keep`
 * SPAWNED backends remain. Only entries with a live child process count
 * toward the cap or are eligible for cap eviction, and — as before — only
 * entries idle beyond `freshMs` may be evicted under the soft cap (an
 * actively kept-alive pool may exceed the soft cap rather than kill a
 * running session). When the pool exceeds the hard cap (keep + 6), even
 * keepalive-fresh entries are evicted LRU so the resident set is bounded
 * (fix for #105239).
 */
export function selectPoolEvictions<K>(
  entries: Iterable<[K, PoolEvictionEntry]>,
  keep: number,
  now: number,
  freshMs: number
): K[] {
  const spawned = [...entries].filter(([, entry]) => Boolean(entry.process))

  if (spawned.length <= keep) {
    return []
  }

  const evictable = spawned
    .filter(([, entry]) => now - (entry.lastActiveAt || 0) > freshMs)
    .sort((a, b) => (a[1].lastActiveAt || 0) - (b[1].lastActiveAt || 0))

  let removable = spawned.length - Math.max(0, keep)
  const evictions: K[] = []

  for (const [key] of evictable) {
    if (removable <= 0) {
      break
    }

    evictions.push(key)
    removable -= 1
  }

  // Hard cap: even keepalive-fresh backends are evicted LRU when the
  // pinned tier grows beyond keep + HARD_CAP_EXTRA. Without this the pool
  // is unbounded — every profile whose chat was ever opened stays pinned
  // indefinitely via the 60s keepalive touch (#105239).
  const hardKeep = Math.max(0, keep) + HARD_CAP_EXTRA
  const remaining = spawned.length - evictions.length

  if (remaining > hardKeep) {
    const evictedSet = new Set(evictions)
    const candidates = spawned
      .filter(([key]) => !evictedSet.has(key))
      .sort((a, b) => (a[1].lastActiveAt || 0) - (b[1].lastActiveAt || 0))

    let need = remaining - hardKeep

    for (const [key] of candidates) {
      if (need <= 0) break
      evictions.push(key)
      need -= 1
    }
  }

  return evictions
}
