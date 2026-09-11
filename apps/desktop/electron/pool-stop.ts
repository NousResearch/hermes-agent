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

export interface PoolStopEntry<Process = unknown> {
  process?: Process
}

export interface PoolStopperDeps<Process> {
  /** The live backend pool. Entries are evicted synchronously on stop. */
  pool: Map<string, PoolStopEntry<Process>>
  /** Signal the child (tree/group kill per platform). Synchronous. */
  stopChild: (child: Process | undefined) => void
  /** Bounded wait: resolves when the child exits, escalating to SIGKILL. */
  waitForExit: (child: Process | undefined) => Promise<void>
}

export interface PoolStopper {
  /** The in-flight stop for a key, if any — await before respawning it. */
  inFlight: (key: string) => Promise<void> | undefined
  /** Whether the pool has a local child or an already-evicted stop in flight. */
  hasPending: () => boolean
  /** Stop one pooled backend; concurrent calls share the same promise. */
  stop: (key: string) => Promise<void>
  /** Stop every pooled backend and join stops already in flight. */
  stopAll: () => Promise<void>
}

interface PendingStop<Process> {
  entry: PoolStopEntry<Process>
  completion: Promise<void>
  failed: boolean
}

export function createPoolStopper<Process>(deps: PoolStopperDeps<Process>): PoolStopper {
  const stops = new Map<string, PendingStop<Process>>()

  function stop(key: string): Promise<void> {
    const inFlight = stops.get(key)

    if (inFlight && !inFlight.failed) {
      return inFlight.completion
    }

    const entry = inFlight?.entry ?? deps.pool.get(key)

    if (!entry) {
      return Promise.resolve()
    }

    // Evict now: routing must not hand out a dying backend. The stop promise
    // below retains the process handle until the bounded exit completes.
    deps.pool.delete(key)

    const stopping = (async (): Promise<void> => {
      deps.stopChild(entry.process)
      await deps.waitForExit(entry.process)
    })().then(
      (): void => {
        stops.delete(key)
      },
      (error: unknown): never => {
        pending.failed = true
        throw error
      }
    )

    const pending: PendingStop<Process> = { entry, completion: stopping, failed: false }
    stops.set(key, pending)

    return stopping
  }

  return {
    inFlight: (key: string): Promise<void> | undefined => stops.get(key)?.completion,
    hasPending: (): boolean =>
      stops.size > 0 || [...deps.pool.values()].some((entry: PoolStopEntry<Process>): boolean => entry.process != null),
    stop,
    stopAll: async (): Promise<void> => {
      const pending = new Set([...deps.pool.keys(), ...stops.keys()])
      const results = await Promise.allSettled([...pending].map(stop))

      const errors = results
        .filter((result: PromiseSettledResult<void>): result is PromiseRejectedResult => result.status === 'rejected')
        .map((result: PromiseRejectedResult): unknown => result.reason)

      if (errors.length) {
        throw new AggregateError(errors, 'Backend pool shutdown failed')
      }
    }
  }
}
