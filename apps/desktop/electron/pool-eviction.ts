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

/**
 * Pick which pool keys the LRU cap should evict so that at most `keep`
 * SPAWNED backends remain. Only entries with a live child process count
 * toward the cap or are eligible for cap eviction, and — as before — only
 * entries idle beyond `freshMs` may be evicted (an actively kept-alive pool
 * may exceed the soft cap rather than kill a running session).
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

  return evictions
}

/**
 * Demand displacement victim for the hard local-backend spawn cap (#102163).
 *
 * selectPoolEvictions() spares keepalive-fresh backends, so a fast A→B→C→D
 * profile switch converges nothing and the 4th spawn queued behind full
 * slots until its ticket expired ("timed out while waiting for a free
 * slot"). A real profile switch outranks background pressure: displace the
 * stalest RUNNING backend (its session persists; reopening respawns) so the
 * pool converges instead of erroring. Only entries holding a child process
 * free a coordinator slot — starting/queued/descriptor entries are never
 * victims, nor is the incoming spawn itself (`exclude`).
 */
export function selectSlotDisplacementVictim<K>(
  entries: Iterable<[K, PoolEvictionEntry]>,
  exclude: K
): K | null {
  let victim: K | null = null
  let victimAt = Number.POSITIVE_INFINITY

  for (const [key, entry] of entries) {
    if (key === exclude || !entry.process) {
      continue
    }

    const at = typeof entry.lastActiveAt === 'number' ? entry.lastActiveAt : 0

    if (at < victimAt) {
      victim = key
      victimAt = at
    }
  }

  return victim
}
