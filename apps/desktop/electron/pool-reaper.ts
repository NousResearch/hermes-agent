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
  lastStreamedAt?: number | null
}

export interface IdleReapDecision {
  profile: string
  idleMs: number
}

/**
 * Pure partition of pool entries into "should be reaped now" vs spared.
 * Entries without a local child process are never idle-reaped.
 *
 * Pinned-tier TTL (#105239): the keepalive refreshes lastActiveAt for every
 * open chat, so that clock alone never fires for the pinned tier. A local
 * child whose last STREAMED turn is older than `pinnedIdleMs` is idle even
 * while keepalive-fresh; entries without the stamp keep the legacy
 * lastActiveAt clock.
 */
export function partitionIdleReapable(
  entries: Iterable<[string, PoolReaperEntry]>,
  now: number,
  idleMs: number,
  pinnedIdleMs?: number
): { reap: IdleReapDecision[]; sparedRemote: string[] } {
  const reap: IdleReapDecision[] = []
  const sparedRemote: string[] = []

  for (const [profile, entry] of entries) {
    if (!entry.process) {
      sparedRemote.push(profile)

      continue
    }

    const idleFor = now - (entry.lastActiveAt || 0)
    const streamedIdleFor = entry.lastStreamedAt ? now - entry.lastStreamedAt : null

    const reapable =
      idleFor > idleMs || (pinnedIdleMs !== undefined && streamedIdleFor !== null && streamedIdleFor > pinnedIdleMs)

    if (reapable) {
      reap.push({ profile, idleMs: Math.round(idleFor / 1000) })
    }
  }

  return { reap, sparedRemote }
}

/*
 * LRU soft-cap accounting is deliberately NOT here: `createPoolRetirer.evictTo`
 * (pool-retire.ts) owns the cap since 9e062912 — it already excludes
 * process-less descriptors from both the candidate set and the budget.
 */
