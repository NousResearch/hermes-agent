/**
 * pool-stop.ts
 *
 * Bounded, deduplicated teardown for pooled profile backends.
 *
 * Idle reaping, LRU eviction, profile deletion, and app quit can all ask to
 * stop the same pooled backend, and historically each caller SIGTERM'd the
 * child and immediately deleted the pool entry. A child that did not exit
 * promptly lost its only handle and survived detached under PID 1 — a live
 * installation accumulated 41 orphaned profile backends this way.
 *
 * createPoolStopper() gives every caller the same contract instead:
 *  - the pool entry is evicted immediately (no router hands out a dying
 *    backend), but the stop promise keeps the process handle until the
 *    bounded SIGTERM -> SIGKILL escalation in waitForExit resolves;
 *  - concurrent stop requests for one key share the single in-flight stop;
 *  - spawn paths can await inFlight(key) so a fresh child never overlaps a
 *    dying one on the same HERMES_HOME.
 *
 * Extracted into a dependency-free module (same pattern as backend-child.ts /
 * pool-eviction.ts) so the dedup and handle-retention semantics are asserted
 * directly instead of grepping main.ts source text.
 */

export interface PoolStopEntry {
  process?: unknown
}

export interface PoolStopperDeps {
  /** The live backend pool. Entries are evicted synchronously on stop. */
  pool: Map<string, PoolStopEntry>
  /** Signal the child (tree/group kill per platform). Synchronous. */
  stopChild: (child: unknown) => void
  /** Bounded wait: resolves when the child exits, escalating to SIGKILL. */
  waitForExit: (child: unknown) => Promise<void>
}

export interface PoolStopper {
  /** The in-flight stop for a key, if any — await before respawning it. */
  inFlight: (key: string) => Promise<void> | undefined
  /** Stop one pooled backend; concurrent calls share the same promise. */
  stop: (key: string) => Promise<void>
  /** Stop every pooled backend currently in the pool. */
  stopAll: () => Promise<void>
}

export function createPoolStopper(deps: PoolStopperDeps): PoolStopper {
  const stops = new Map<string, Promise<void>>()

  function stop(key: string): Promise<void> {
    const inFlight = stops.get(key)

    if (inFlight) {
      return inFlight
    }

    const entry = deps.pool.get(key)

    if (!entry) {
      return Promise.resolve()
    }

    // Evict now: routing must not hand out a dying backend. The stop promise
    // below retains the process handle until the bounded exit completes.
    deps.pool.delete(key)

    const stopping = (async () => {
      deps.stopChild(entry.process)
      await deps.waitForExit(entry.process)
    })().finally(() => {
      stops.delete(key)
    })

    stops.set(key, stopping)

    return stopping
  }

  return {
    inFlight: key => stops.get(key),
    stop,
    stopAll: async () => {
      await Promise.all([...deps.pool.keys()].map(stop))
    }
  }
}
