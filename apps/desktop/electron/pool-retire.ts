import type { LocalBackendSpawnCoordinator } from './pool-spawn-coordinator'

export interface PoolRetireEntry {
  /** An early veto only; backend admission owns the proof. */
  activeTurn?: boolean
  lastActiveAt?: null | number
  process?: unknown
  /** A pooled descriptor entry (registry/remote) has no child but IS a live backend. */
  connectionPromise?: null | Promise<unknown>
}

/**
 * Whether a stale pool entry has a backend whose admission fence can be
 * addressed at all — a local child at its own port, a descriptor at the backend
 * behind its resolved connection. An entry whose connect never reached a backend
 * has no fence to ask and stays reclaimable on staleness alone.
 */
function fenceable(entry: PoolRetireEntry): boolean {
  return Boolean(entry.process || entry.connectionPromise)
}

/**
 * Whether tearing this entry down stops a backend process, which is what makes
 * a work proof mandatory before the teardown.
 *
 * A local child is ours. A remote `serve` this app spawned over SSH is ours too
 * — its teardown kills the remote child — so a descriptor entry MUST be passed
 * `ownedRemote` from the caller that knows it. A plain remote descriptor is only
 * a local handle: dropping it kills nothing, and the fence behind
 * ``/api/health/retirement`` is irreversible, so fencing a backend we never stop
 * would strand it.
 */
export function teardownStopsBackend(entry: PoolRetireEntry, ownedRemote: boolean): boolean {
  return Boolean(entry.process) || ownedRemote
}

export interface PoolRetirerDeps<E extends PoolRetireEntry> {
  pool: Map<string, E>
  coordinator: LocalBackendSpawnCoordinator
  prepare: (key: string, entry: E) => Promise<string | null>
  commit: (key: string, entry: E, token: string) => Promise<boolean>
  cancel: (key: string, entry: E, token: string) => Promise<void>
  stopBackend: (key: string) => Promise<void>
  onRetiring?: (key: string) => void
  log?: (message: string) => void
}

export function selectRetirementCandidates<K, E extends PoolRetireEntry>(
  entries: Iterable<[K, E]>,
  exclude: ReadonlySet<K>
): [K, E][] {
  return [...entries]
    .filter(([key, entry]) => Boolean(entry.process) && entry.activeTurn !== true && !exclude.has(key))
    .sort((a, b) => (a[1].lastActiveAt || 0) - (b[1].lastActiveAt || 0))
}

/** One arbiter owns foreground recovery, idle reaping and stale LRU eviction.
 * A permit freezes backend admission; a GET snapshot never authorizes a kill.
 * Credit: @bounce12340's occupied-not-busy diagnosis, @austinpickett's parking,
 * @chelsealong's backend-work proof and @sharkenstein3d's shared stop boundary.
 */
export function createPoolRetirer<E extends PoolRetireEntry>(deps: PoolRetirerDeps<E>) {
  let serial: Promise<unknown> = Promise.resolve()
  let scheduled = false
  let disposed = false
  const retiredScopes = new Set<string>()
  let retry: ReturnType<typeof setTimeout> | undefined
  const report = (error: unknown) => deps.log?.(`Pool retirement failed: ${String(error)}`)

  function enqueue<T>(work: () => Promise<T>): Promise<T> {
    const result = serial.then(work)
    serial = result.catch(report)

    return result
  }

  const needsCapacity = () => !disposed && deps.coordinator.foregroundWaiters.size > 0 &&
    deps.coordinator.activeCount >= deps.coordinator.limit

  async function retire(key: string, entry: E, needed: () => boolean): Promise<boolean> {
    const eligible = () => !disposed && deps.pool.get(key) === entry && entry.activeTurn !== true && needed()

    // Both pool shapes route through one admission authority: a local child
    // proves work at its own port, a descriptor entry at the backend behind
    // its resolved connection. Nothing is torn down on staleness alone.
    if (!fenceable(entry) || !eligible()) {
      return false
    }

    const token = await deps.prepare(key, entry)

    if (!token) {
      return false
    }

    let committed = false

    try {
      // A different resident may have exited, the waiter withdrawn, or this
      // key acquired a new generation while the HTTP prepare was outstanding.
      if (!eligible()) {
        return false
      }

      committed = await deps.commit(key, entry, token)

      if (!committed || deps.pool.get(key) !== entry) {
        return false
      }

      // Commit is irreversible: admission remains closed even if capacity or
      // renderer intent changes now. Park before signalling the exact child.
      retiredScopes.add(key)
      deps.onRetiring?.(key)
      await deps.stopBackend(key)

      return true
    } finally {
      if (!committed) {
        await deps.cancel(key, entry, token).catch(report)
      }
    }
  }

  async function reclaim(): Promise<void> {
    while (needsCapacity()) {
      let retired = false
      const candidates = selectRetirementCandidates(deps.pool, deps.coordinator.foregroundWaiters)

      for (const [key, entry] of candidates) {
        if (!needsCapacity()) {
          return
        }

        if (await retire(key, entry, needsCapacity)) {
          retired = true

          break
        }
      }

      if (!retired) {
        return
      }
    }
  }

  function wake(): void {
    if (scheduled || disposed) {
      return
    }

    clearTimeout(retry)
    scheduled = true
    void enqueue(reclaim).catch(report).finally(() => {
      scheduled = false

      // Busy work may finish without a renderer touch (cron / side agents).
      // The coordinator's existing ticket deadline bounds these retries.
      if (needsCapacity()) {
        retry = setTimeout(wake, 1000)
        retry.unref?.()
      }
    })
  }

  const unsubscribe = deps.coordinator.onChange(wake)

  return {
    wake,
    assertCanOpen: (key: string, priority: 'foreground' | 'background') => {
      if (priority === 'foreground') {
        retiredScopes.delete(key)
      } else if (retiredScopes.has(key)) {
        throw new Error(`Backend for "${key}" was retired; open it explicitly to reconnect.`)
      }
    },
    retireIdle: (key: string, idleMs: number) => enqueue(async () => {
      const entry = deps.pool.get(key)

      return entry ? retire(key, entry, () => Date.now() - (entry.lastActiveAt || 0) > idleMs) : false
    }),
    evictTo: (keep: number, freshMs: number) => enqueue(async () => {
      const retired: string[] = []
      const overCap = () => [...deps.pool.values()].filter(entry => entry.process).length > Math.max(0, keep)

      for (const [key, entry] of selectRetirementCandidates(deps.pool, deps.coordinator.foregroundWaiters)) {
        if (await retire(key, entry, () => overCap() && Date.now() - (entry.lastActiveAt || 0) > freshMs)) {
          retired.push(key)
        }
      }

      return retired
    }),
    dispose: () => {
      disposed = true
      clearTimeout(retry)
      unsubscribe()
    },
  }
}
