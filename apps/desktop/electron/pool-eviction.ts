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
  /** True for a local backend reservation before its child process is attached. */
  countsTowardPoolCap?: boolean
  /**
   * Number of bounded local worker slots reserved by this entry. Omitted means
   * one slot, preserving ordinary profile-backend behavior. Only Electron may
   * create a weighted reservation.
   */
  capacityUnits?: number
}

export const POOL_CAPACITY_EXCEEDED = 'POOL_CAPACITY_EXCEEDED'

/** Zero is the explicit operator-selected unlimited mode. */
export function isUnlimitedPoolCapacity(max: number): boolean {
  return max === 0
}

export class PoolCapacityError extends Error {
  readonly code = POOL_CAPACITY_EXCEEDED

  constructor(max: number) {
    super(`Local profile backend budget reached (${max}); wait for an active profile backend to finish or close it before opening another.`)
    this.name = 'PoolCapacityError'
  }
}

function isLocalBackendEntry(entry: PoolEvictionEntry): boolean {
  return Boolean(entry.process) || entry.countsTowardPoolCap === true
}

function capacityUnits(entry: PoolEvictionEntry): number {
  if (!isLocalBackendEntry(entry)) {
    return 0
  }

  return Number.isInteger(entry.capacityUnits) && (entry.capacityUnits as number) > 0
    ? (entry.capacityUnits as number)
    : 1
}

/**
 * Hard admission guard for local profile backends. Unlike LRU eviction this
 * counts in-flight reservations, so concurrent profile opens cannot all observe
 * spare capacity and stampede the host with MCP-heavy child trees.
 */
export function canAdmitLocalBackend(
  entries: Iterable<[unknown, PoolEvictionEntry]>,
  max: number,
  requestedCapacity = 1
): boolean {
  if (!Number.isInteger(requestedCapacity) || requestedCapacity < 1) {
    return false
  }

  if (isUnlimitedPoolCapacity(max)) {
    return true
  }

  const maximum = Math.max(1, max)
  if (requestedCapacity > maximum) {
    return false
  }

  const used = [...entries].reduce((total, [, entry]) => total + capacityUnits(entry), 0)
  return used + requestedCapacity <= maximum
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
  const spawned = [...entries].filter(([, entry]) => isLocalBackendEntry(entry))
  const usedCapacity = spawned.reduce((total, [, entry]) => total + capacityUnits(entry), 0)

  if (usedCapacity <= keep) {
    return []
  }

  const evictable = spawned
    .filter(([, entry]) => now - (entry.lastActiveAt || 0) > freshMs)
    .sort((a, b) => (a[1].lastActiveAt || 0) - (b[1].lastActiveAt || 0))

  let removableCapacity = usedCapacity - Math.max(0, keep)
  const evictions: K[] = []

  for (const [key, entry] of evictable) {
    if (removableCapacity <= 0) {
      break
    }

    evictions.push(key)
    removableCapacity -= capacityUnits(entry)
  }

  return evictions
}
