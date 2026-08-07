/**
 * Pool reaper policy: which pooled profile backends the idle timer and the
 * LRU soft cap may stop.
 *
 * A pool entry is either a LOCAL child process (`process` set — it costs RAM
 * and holds a venv lock, so idling it out is the point of the reaper) or a
 * REMOTE descriptor (`process === null` — stopping it only deletes a cheap
 * Map entry, but the renderer's next message has to rebuild the whole
 * connection, which surfaces to the user as "my parallel session vanished").
 *
 * Remote descriptors already have a death policy: `revalidatePooledRemoteBackends`
 * probes `/api/status` and drops descriptors for genuinely unreachable hosts.
 * They must NOT also be subject to an *idle* timer — an idle-but-healthy
 * remote backend is indistinguishable from a dead one to a timer, and the
 * failure mode (silently dropped session) is far worse than the cost of
 * keeping the descriptor (a few hundred bytes).
 */

export interface PoolReaperEntry {
  process?: unknown
  lastActiveAt?: number | null
}

export interface IdleReapDecision {
  profile: string
  idleMs: number
}

/**
 * Pure partition of pool entries into "should be reaped now" vs spared.
 * Entries without a local child process are never idle-reaped.
 */
export function partitionIdleReapable(
  entries: Iterable<[string, PoolReaperEntry]>,
  now: number,
  idleMs: number
): { reap: IdleReapDecision[]; sparedRemote: string[] } {
  const reap: IdleReapDecision[] = []
  const sparedRemote: string[] = []

  for (const [profile, entry] of entries) {
    if (!entry.process) {
      sparedRemote.push(profile)
      continue
    }

    const idleFor = now - (entry.lastActiveAt || 0)

    if (idleFor > idleMs) {
      reap.push({ profile, idleMs: Math.round(idleFor / 1000) })
    }
  }

  return { reap, sparedRemote }
}

/**
 * LRU eviction candidates for the soft cap. Remote descriptors are excluded
 * for the same reason as above; among local backends only entries whose
 * renderer socket went stale (past `keepaliveFreshMs`) may be evicted, so a
 * live multi-profile session can exceed the soft cap rather than abort a
 * running agent.
 */
export function selectLruEvictionCandidates(
  entries: Iterable<[string, PoolReaperEntry]>,
  now: number,
  keepaliveFreshMs: number
): string[] {
  return [...entries]
    .filter(([, entry]) => Boolean(entry.process))
    .filter(([, entry]) => now - (entry.lastActiveAt || 0) > keepaliveFreshMs)
    .sort((a, b) => (a[1].lastActiveAt || 0) - (b[1].lastActiveAt || 0))
    .map(([profile]) => profile)
}

/**
 * Count of pool entries that hold a local child process. The LRU eviction
 * budget must be based on this — not on the total pool size — so that cheap
 * remote descriptors never push a still-needed local backend over the cap in a
 * mixed pool. (Regression: several remote descriptors + one local backend below
 * the local cap must yield a budget of 0, evicting nothing.)
 */
export function countLocalBackends(entries: Iterable<[string, PoolReaperEntry]>): number {
  let count = 0
  for (const [, entry] of entries) {
    if (entry && entry.process) {
      count += 1
    }
  }
  return count
}
