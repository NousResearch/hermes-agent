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
 * Evict enough stale spawned backends to make room, and do not report the
 * room as available until every selected child has actually exited.
 *
 * Selection and teardown deliberately live in one helper: callers that fire
 * stopBackend() without awaiting it can enqueue a replacement while the old
 * child still owns its hard spawn slot. Under a restore stampede that race
 * turns successful LRU selection into a 30-second queue timeout.
 */
export async function evictPoolEntries<K>(
  entries: Iterable<[K, PoolEvictionEntry]>,
  keep: number,
  now: number,
  freshMs: number,
  stopBackend: (key: K) => Promise<void>
): Promise<K[]> {
  const evictions = selectPoolEvictions(entries, keep, now, freshMs)

  await Promise.all(evictions.map(key => stopBackend(key)))

  return evictions
}
