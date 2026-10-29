/**
 * registry-session-enumerator.ts
 *
 * Bounded background enumeration of registry connections, feeding the
 * federated session index (federated-session-index.ts).
 *
 * The unified Sessions list reads only already-pooled backends (main.ts
 * pooledRegistrySessionSources: a sidebar refresh must never dial or spawn a
 * backend), so a connection without a warm backend contributes zero rows with
 * no distinction between "no conversations" and "not fetched". This scheduler
 * is the one separate lane that is allowed to dial: it walks the registry's
 * non-local connections on a slow cycle and hands each owner-tagged read to the
 * index, so a cold connection's history is still there when the sidebar needs it.
 *
 * It only runs for registries holding more than one connection — with a single
 * connection the primary backend already serves every row through the normal
 * aggregate path, so there is nothing to enumerate and behaviour is unchanged.
 * The local connection and the registry primary are never enumerated for the
 * same reason.
 *
 * Bounded in every dimension:
 *
 *  - nothing runs for INITIAL_DELAY_MS after start(), so enumeration never
 *    competes with the foreground boot for the machine or the network
 *  - one in-flight pass per connection, and one pass at a time across the
 *    registry, each STAGGER_MS apart
 *  - one pass per connection per interval; a connection whose pass failed backs
 *    off exponentially (up to MAX_BACKOFF_MS) so a dead host is not re-dialed
 *    every cycle
 *  - a pass never throws into the host: errors are swallowed and counted
 *
 * Kept electron-free (no `import 'electron'`) so it unit-tests under the
 * `electron` vitest project; main.ts injects the dial/fetch/persist callbacks.
 */

/** A registry connection a pass may cover. */
export interface EnumerationTarget {
  connectionId: string
  /** `local` is never enumerated; ssh/remote/cloud are. */
  kind: string
  /** The registry primary owns the window backend, whose rows the primary
   * aggregate always carries — never enumerated. */
  primary?: boolean
}

export const ENUMERATION_INTERVAL_MS = 5 * 60 * 1000
/** Held back after start() so the foreground boot is never contested. */
export const ENUMERATION_INITIAL_DELAY_MS = 30_000
/** Rows asked of one backend per pass: one page, comparable to a sidebar refresh. */
export const ENUMERATION_PAGE_LIMIT = 100
export const ENUMERATION_FETCH_TIMEOUT_MS = 10_000
/** Gap between two passes, so two connections never dial back to back. */
export const STAGGER_MS = 5_000
/** A failing connection is retried after `interval * 2^failures`, capped here. */
export const MAX_BACKOFF_MS = 60 * 60 * 1000

/** Host contract injected by main.ts. Every leg is bounded by the caller; the
 * enumerator only owns scheduling and locking. */
export interface EnumerationHost {
  /** The registered connections (main.ts: readDesktopConnectionsRegistry). */
  connections: () => EnumerationTarget[]
  /**
   * Read one connection's cross-profile session rows, dialing if no backend is
   * pooled. Returns the owner-tagged rows on success and null on any failure
   * (dead host, auth failure, timeout, malformed reply). Empty rows are a
   * successful read.
   */
  fetchSessions: (connectionId: string) => Promise<{ sessions: unknown[] } | null>
  /** Persist a successful read into the federated index. */
  remember: (connectionId: string, sessions: unknown[]) => void
  /** Called with the registered connection ids at the start of every cycle, so
   * the index can drop connections that left the registry. */
  retain?: (connectionIds: ReadonlySet<string>) => void
  /** setTimeout/clearTimeout pair — injectable for tests. */
  schedule: (fn: () => void, ms: number) => unknown
  cancel: (handle: unknown) => void
  /** Test seam; defaults to Date.now(). */
  now?: () => number
}

export interface EnumeratorOptions {
  initialDelayMs?: number
  intervalMs?: number
  staggerMs?: number
}

/**
 * The scheduler. One instance lives in Electron main; `start()` is idempotent
 * and `stop()` fully tears down (quit).
 */
export class RegistrySessionEnumerator {
  private readonly host: EnumerationHost
  private readonly initialDelayMs: number
  private readonly intervalMs: number
  private readonly staggerMs: number
  private readonly queue: string[] = []
  private readonly inFlight = new Set<string>()
  private readonly lastPassAt = new Map<string, number>()
  private readonly failures = new Map<string, number>()
  private running = false
  private timer: unknown = null
  private drainTimer: unknown = null

  constructor(host: EnumerationHost, options: EnumeratorOptions = {}) {
    this.host = host
    this.initialDelayMs = options.initialDelayMs ?? ENUMERATION_INITIAL_DELAY_MS
    this.intervalMs = options.intervalMs ?? ENUMERATION_INTERVAL_MS
    this.staggerMs = options.staggerMs ?? STAGGER_MS
  }

  /** Whether a pass is currently executing (unit-test observable). */
  get busy(): boolean {
    return this.inFlight.size > 0
  }

  start(): void {
    if (this.running) {
      return
    }

    this.running = true
    this.timer = this.host.schedule(this.tickLoop, this.initialDelayMs)
  }

  stop(): void {
    this.running = false
    this.queue.length = 0

    if (this.timer !== null) {
      this.host.cancel(this.timer)
      this.timer = null
    }

    if (this.drainTimer !== null) {
      this.host.cancel(this.drainTimer)
      this.drainTimer = null
    }
  }

  private now(): number {
    return this.host.now?.() ?? Date.now()
  }

  /** How long a connection waits between passes: the interval, doubled per
   * consecutive failure, capped. */
  private waitFor(connectionId: string): number {
    const failed = this.failures.get(connectionId) ?? 0

    return Math.min(this.intervalMs * 2 ** failed, Math.max(MAX_BACKOFF_MS, this.intervalMs))
  }

  /** One scheduling decision: enqueue every due, idle, non-local connection,
   * then drain. A registry with fewer than two connections is skipped
   * entirely — the primary aggregate already covers that topology. */
  private tick = (): void => {
    const targets = this.host.connections()

    this.host.retain?.(new Set(targets.map(target => target.connectionId)))

    if (targets.length < 2) {
      return
    }

    const now = this.now()

    for (const target of targets) {
      if (target.kind === 'local' || target.primary) {
        continue
      }

      const last = this.lastPassAt.get(target.connectionId)

      if (last !== undefined && now - last < this.waitFor(target.connectionId)) {
        continue
      }

      if (this.inFlight.has(target.connectionId) || this.queue.includes(target.connectionId)) {
        continue
      }

      this.queue.push(target.connectionId)
    }

    this.drain()
  }

  private tickLoop = (): void => {
    if (!this.running) {
      return
    }

    this.tick()
    this.timer = this.host.schedule(this.tickLoop, this.intervalMs)
  }

  /** Serialized staggered drain: at most one pass executes at a time; the next
   * is scheduled staggerMs later. A connection removed from the registry
   * between enqueue and execution is skipped. */
  private drain = (): void => {
    if (!this.running || this.drainTimer !== null) {
      return
    }

    const next = this.queue.shift()

    if (next === undefined) {
      return
    }

    if (!this.host.connections().some(target => target.connectionId === next)) {
      this.drain()

      return
    }

    this.inFlight.add(next)
    this.lastPassAt.set(next, this.now())

    const settle = () => {
      this.inFlight.delete(next)

      if (this.running && this.queue.length) {
        this.drainTimer = this.host.schedule(() => {
          this.drainTimer = null
          this.drain()
        }, this.staggerMs)
      }
    }

    void this.host
      .fetchSessions(next)
      .then(result => {
        if (!result) {
          this.failures.set(next, (this.failures.get(next) ?? 0) + 1)

          return
        }

        this.failures.delete(next)
        this.host.remember(next, result.sessions)
      })
      .catch(() => {
        this.failures.set(next, (this.failures.get(next) ?? 0) + 1)
      })
      .finally(settle)
  }
}
