export const REMOTE_LIVENESS_TIMEOUT_MS = 10_000
// Dispatch is synchronous user intent: a cached descriptor must prove its
// forwarded endpoint is alive before it can be returned. Keep this probe much
// shorter than the background liveness budget so a dead tunnel reconnects
// promptly instead of making the click feel hung.
export const POOLED_REMOTE_DISPATCH_PROBE_TIMEOUT_MS = 2_500
export const REMOTE_LIVENESS_FAILURE_LIMIT = 3
// Even at the capped retry path, consecutive liveness observations are at most
// about 48s apart (ticket mint + socket open + backoff + the next status probe).
// One minute keeps a continuous outage together without carrying old failures.
export const REMOTE_LIVENESS_FAILURE_WINDOW_MS = 60_000

// A dispatch-probe miss is per-profile evidence, but the cause is often not.
// When one remote host is briefly saturated, MANY pooled profiles on the SAME
// connection miss their probe inside the same tick (measured: 21 profiles
// within 40ms). Treating each miss independently retires and re-dials all of
// them at once, which lands N SSH handshakes + N remote backend boots on the
// host that was already too busy to answer — the misses cause the stampede
// that causes the next round of misses. These constants classify that shape:
// distinct PROFILES (never one profile failing repeatedly) failing on one
// connection inside a short window is a host event, not N dead backends.
export const HOST_EVENT_DISTINCT_PROFILE_THRESHOLD = 3
export const HOST_EVENT_WINDOW_MS = 2_000
// Long enough for a load spike to drain, short enough that a genuinely dead
// backend still reconnects well inside the user's patience for one click.
export const HOST_EVENT_BACKOFF_MS = 5_000

export interface RemoteLivenessFailure {
  failures: number
  shouldReset: boolean
}

interface RemoteConnectionDescriptor {
  baseUrl?: null | string
  mode?: null | string
}

/**
 * Last look before the primary connection is dropped.
 *
 * The pooled path can recognise a busy host because many profiles miss at
 * once. The primary connection is alone on its gateway, so it has no siblings
 * to vote with — its only evidence is its own streak, and on a saturated host
 * that streak expires in a few seconds of quick probes. Dropping it is a
 * visible whole-app reload, so it is worth one pause plus one recheck first.
 */
export interface RemoteDropConfirmation {
  backoff: () => Promise<void>
  /** Cheap transport-level check (SSH master liveness), when one applies. */
  transportAlive?: () => Promise<boolean>
}

export interface RevalidateRemoteConnectionOptions<TConnection extends RemoteConnectionDescriptor> {
  confirmDrop?: RemoteDropConfirmation
  connectionPromise: Promise<TConnection>
  currentConnectionPromise: () => null | Promise<TConnection>
  log: (message: string) => void
  probe: (connection: TConnection, path: string, options: { timeoutMs: number }) => Promise<unknown>
  resetConnection: () => void
  tracker: RemoteLivenessTracker
}

export interface RemoteRevalidationResult {
  ok: true
  rebuilt: boolean
}

/**
 * Coalesces revalidation work for one cached connection promise.
 *
 * Every Desktop BrowserWindow owns a renderer gateway loop. When several
 * windows observe the same disconnect they can all ask the Electron main
 * process to revalidate the shared primary connection at once. Those calls
 * must count as one probe, not several consecutive failures.
 */
export class RemoteRevalidationCoordinator {
  readonly #inflightByConnection = new WeakMap<object, Promise<unknown>>()

  run<T>(connection: object, task: () => Promise<T>): Promise<T> {
    const existing = this.#inflightByConnection.get(connection) as Promise<T> | undefined

    if (existing) {
      return existing
    }

    const pending = Promise.resolve().then(task)

    const clear = () => {
      if (this.#inflightByConnection.get(connection) === pending) {
        this.#inflightByConnection.delete(connection)
      }
    }

    this.#inflightByConnection.set(connection, pending)
    // Clean up on both outcomes without creating an unhandled rejected branch.
    void pending.then(clear, clear)

    return pending
  }
}

/**
 * Classifies correlated dispatch-probe failures on one connection as a HOST
 * EVENT (the host is busy) rather than N independent dead backends.
 *
 * Keyed per connection id, counting DISTINCT pool keys inside a rolling
 * window: one profile failing three times in a row is still one dead backend
 * and must keep the existing fast retire path. Failures age out of the window
 * on their own, so the classification is self-clearing — there is no latch to
 * reset and no timer to own.
 */
export class RemoteHostEventTracker {
  readonly #failuresByConnection = new Map<string, Map<string, number>>()
  readonly #now: () => number
  readonly #threshold: number
  readonly #windowMs: number

  constructor(
    threshold = HOST_EVENT_DISTINCT_PROFILE_THRESHOLD,
    windowMs = HOST_EVENT_WINDOW_MS,
    now: () => number = Date.now
  ) {
    if (!Number.isInteger(threshold) || threshold < 2) {
      throw new Error('Host event threshold must be an integer of at least 2.')
    }

    if (!Number.isFinite(windowMs) || windowMs < 1) {
      throw new Error('Host event window must be positive.')
    }

    this.#threshold = threshold
    this.#windowMs = windowMs
    this.#now = now
  }

  /**
   * Record one dispatch-probe failure. Returns true once enough DISTINCT pool
   * keys on this connection have failed inside the window — including for the
   * failure that crosses the threshold and every later one while it holds.
   */
  recordProbeFailure(connectionId: string, poolKey: string): boolean {
    const now = this.#now()
    let recent = this.#failuresByConnection.get(connectionId)

    if (!recent) {
      recent = new Map()
      this.#failuresByConnection.set(connectionId, recent)
    }

    recent.set(poolKey, now)

    for (const [key, at] of recent) {
      if (now - at > this.#windowMs) {
        recent.delete(key)
      }
    }

    if (recent.size === 0) {
      this.#failuresByConnection.delete(connectionId)
    }

    return recent.size >= this.#threshold
  }

  clear(connectionId?: string): void {
    if (connectionId === undefined) {
      this.#failuresByConnection.clear()

      return
    }

    this.#failuresByConnection.delete(connectionId)
  }
}

// Recovery must not become the next stampede: three in flight keeps a busy
// host making progress without handing it 20+ handshakes at once, and a small
// jitter stops each released batch from landing on the same millisecond.
export const POOLED_REMOTE_DIAL_CONCURRENCY = 3
export const POOLED_REMOTE_DIAL_JITTER_MS = 250

export interface PooledRemoteDialGateOptions {
  delay?: (ms: number) => Promise<void>
  jitterMs?: number
  limit?: number
  random?: () => number
}

/**
 * Caps how many pooled remote dials for ONE connection id may be in flight.
 *
 * BackendDialClaims already coalesces concurrent dials of a single pool key;
 * this is the orthogonal bound across the keys that share a host. A caller
 * that finds a free slot runs immediately and pays nothing, so single-profile
 * reconnect stays as fast as it was; only callers that would have exceeded the
 * cap wait, and they are admitted one at a time with jitter as slots free.
 */
export class PooledRemoteDialGate {
  readonly #delay: (ms: number) => Promise<void>
  readonly #jitterMs: number
  readonly #lanes = new Map<string, { active: number; queue: Array<() => void> }>()
  readonly #limit: number
  readonly #random: () => number

  constructor({
    delay = ms => new Promise<void>(resolve => setTimeout(resolve, ms)),
    jitterMs = POOLED_REMOTE_DIAL_JITTER_MS,
    limit = POOLED_REMOTE_DIAL_CONCURRENCY,
    random = Math.random
  }: PooledRemoteDialGateOptions = {}) {
    if (!Number.isInteger(limit) || limit < 1) {
      throw new Error('Pooled remote dial concurrency must be a positive integer.')
    }

    if (!Number.isFinite(jitterMs) || jitterMs < 0) {
      throw new Error('Pooled remote dial jitter must be zero or positive.')
    }

    this.#delay = delay
    this.#jitterMs = jitterMs
    this.#limit = limit
    this.#random = random
  }

  /** In-flight dials for this connection id (test/diagnostic seam). */
  active(connectionId: string): number {
    return this.#lanes.get(connectionId)?.active || 0
  }

  async run<T>(connectionId: string, dial: () => Promise<T> | T): Promise<T> {
    let lane = this.#lanes.get(connectionId)

    if (!lane) {
      lane = { active: 0, queue: [] }
      this.#lanes.set(connectionId, lane)
    }

    if (lane.active >= this.#limit) {
      await new Promise<void>(resolve => {
        lane.queue.push(resolve)
      })
    } else {
      lane.active += 1
    }

    try {
      return await dial()
    } finally {
      const next = lane.queue.shift()

      if (next) {
        // Hand the slot straight to the next waiter rather than releasing it:
        // a released slot could be claimed by a fresh caller, letting the
        // waiter's own admission push the lane past the cap.
        void this.#delay(this.#random() * this.#jitterMs).then(next)
      } else {
        lane.active -= 1

        if (lane.active === 0 && this.#lanes.get(connectionId) === lane) {
          this.#lanes.delete(connectionId)
        }
      }
    }
  }
}

export interface DispatchHostEventPolicy {
  /** Sleep out the host-event backoff before revalidating the descriptor. */
  backoff: () => Promise<void>
  /** Record this failure; true when the connection is inside a host event. */
  classify: () => boolean
  /** Cheap host-level liveness (`ssh -O check`); false skips to teardown. */
  hostAlive: () => Promise<boolean>
}

interface EnsureHealthyPooledRemoteBackendForDispatchOptions<TConnection extends RemoteConnectionDescriptor> {
  connectionPromise: Promise<TConnection>
  currentConnectionPromise: () => null | Promise<TConnection>
  hostEvent?: DispatchHostEventPolicy
  log?: (message: string) => void
  probe: (connection: TConnection, path: string, options: { timeoutMs: number }) => Promise<unknown>
  reconnect: () => Promise<TConnection>
  retire: (error: unknown) => Promise<void> | void
}

/**
 * Gate dispatch through a cheap health probe of the exact cached descriptor.
 *
 * A failed descriptor is retired before reconnecting, while identity checks
 * prevent a late probe from tearing down a replacement installed by another
 * caller. The caller should single-flight this function per cached promise so
 * concurrent dispatches share one retire/reconnect sequence.
 *
 * When a `hostEvent` policy is supplied and it classifies this failure as one
 * of many across the connection, teardown is DEFERRED: back off, confirm the
 * host is still reachable, and re-probe the same descriptor. A descriptor that
 * answers is kept as-is — the host was busy, not dead — which is what stops a
 * host hiccup from turning into N teardowns plus N reconnects. A descriptor
 * that still fails falls through to the unchanged retire/reconnect path.
 */
export async function ensureHealthyPooledRemoteBackendForDispatch<TConnection extends RemoteConnectionDescriptor>({
  connectionPromise,
  currentConnectionPromise,
  hostEvent,
  log,
  probe,
  reconnect,
  retire
}: EnsureHealthyPooledRemoteBackendForDispatchOptions<TConnection>): Promise<TConnection> {
  let connection: TConnection
  let descriptorResolved = false

  try {
    connection = await connectionPromise
    descriptorResolved = true

    if (currentConnectionPromise() !== connectionPromise) {
      return reconnect()
    }

    await probe(connection, '/api/status', {
      timeoutMs: POOLED_REMOTE_DISPATCH_PROBE_TIMEOUT_MS
    })
  } catch (error) {
    // A descriptor whose boot never resolved has nothing to revalidate, so the
    // host-event path only applies once a probe of a real descriptor failed.
    if (descriptorResolved && hostEvent && currentConnectionPromise() === connectionPromise && hostEvent.classify()) {
      log?.(
        `Pooled remote backend failed its dispatch probe during a host event; deferring teardown for ${HOST_EVENT_BACKOFF_MS}ms before revalidating.`
      )
      await hostEvent.backoff()

      if (currentConnectionPromise() !== connectionPromise) {
        return reconnect()
      }

      if (await pooledRemoteBackendSurvivedHostEvent(connection!, hostEvent, probe)) {
        log?.('Pooled remote backend answered after the host-event backoff; keeping the descriptor.')

        return connection!
      }
    }

    if (currentConnectionPromise() === connectionPromise) {
      await retire(error)
    }

    return reconnect()
  }

  if (currentConnectionPromise() !== connectionPromise) {
    return reconnect()
  }

  return connection
}

/**
 * Second chance for one descriptor after the host-event backoff: the transport
 * must be up AND the backend must answer. Either miss means this really is a
 * dead backend, so the caller retires it exactly as before.
 */
async function pooledRemoteBackendSurvivedHostEvent<TConnection extends RemoteConnectionDescriptor>(
  connection: TConnection,
  hostEvent: DispatchHostEventPolicy,
  probe: (connection: TConnection, path: string, options: { timeoutMs: number }) => Promise<unknown>
): Promise<boolean> {
  try {
    if (!(await hostEvent.hostAlive())) {
      return false
    }

    await probe(connection, '/api/status', {
      timeoutMs: POOLED_REMOTE_DISPATCH_PROBE_TIMEOUT_MS
    })

    return true
  } catch {
    return false
  }
}

/**
 * Tracks consecutive remote liveness failures independently per gateway.
 * A successful probe clears the streak, and reaching the limit consumes it so
 * a rebuilt connection starts from a clean state.
 */
export class RemoteLivenessTracker {
  readonly #failureLimit: number
  readonly #failureWindowMs: number
  readonly #failuresByBaseUrl = new Map<string, { failures: number; lastFailureAt: number }>()
  readonly #now: () => number

  constructor(
    failureLimit = REMOTE_LIVENESS_FAILURE_LIMIT,
    failureWindowMs = REMOTE_LIVENESS_FAILURE_WINDOW_MS,
    now: () => number = Date.now
  ) {
    if (!Number.isInteger(failureLimit) || failureLimit < 1) {
      throw new Error('Remote liveness failure limit must be a positive integer.')
    }

    if (!Number.isFinite(failureWindowMs) || failureWindowMs < 1) {
      throw new Error('Remote liveness failure window must be positive.')
    }

    this.#failureLimit = failureLimit
    this.#failureWindowMs = failureWindowMs
    this.#now = now
  }

  recordSuccess(baseUrl: string): void {
    this.#failuresByBaseUrl.delete(baseUrl)
  }

  recordFailure(baseUrl: string): RemoteLivenessFailure {
    const now = this.#now()
    const previous = this.#failuresByBaseUrl.get(baseUrl)
    const withinFailureWindow = previous && now - previous.lastFailureAt <= this.#failureWindowMs
    const failures = (withinFailureWindow ? previous.failures : 0) + 1
    const shouldReset = failures >= this.#failureLimit

    if (shouldReset) {
      this.#failuresByBaseUrl.delete(baseUrl)
    } else {
      this.#failuresByBaseUrl.set(baseUrl, { failures, lastFailureAt: now })
    }

    return { failures, shouldReset }
  }

  clear(): void {
    this.#failuresByBaseUrl.clear()
  }
}

export interface PooledRemoteEntry<TConnection extends RemoteConnectionDescriptor = RemoteConnectionDescriptor> {
  connectionPromise?: null | Promise<TConnection>
  process?: unknown
  remoteBaseUrl?: null | string
}

export interface RevalidatePooledRemoteBackendsOptions<TConnection extends RemoteConnectionDescriptor> {
  entries: Iterable<[string, PooledRemoteEntry<TConnection>]>
  log: (message: string) => void
  probe: (connection: TConnection, path: string, options: { timeoutMs: number }) => Promise<unknown>
  stopBackend: (profile: string) => void
  tracker: RemoteLivenessTracker
}

/**
 * Probe pooled REMOTE descriptors and drop the dead ones.
 *
 * A pooled entry backed by a remote host has no child process, so the 'exit'
 * handler that clears a dead local backend never fires, and the renderer's
 * keepalive touch keeps the idle reaper off it. Without this the pool serves a
 * descriptor for an unreachable host indefinitely.
 *
 * Entries share the primary's failure policy, keyed per base URL, so a profile
 * pointing at the same host as another does not burn the streak twice as fast.
 */
export async function revalidatePooledRemoteBackends<TConnection extends RemoteConnectionDescriptor>({
  entries,
  log,
  probe,
  stopBackend,
  tracker
}: RevalidatePooledRemoteBackendsOptions<TConnection>): Promise<{ dropped: string[] }> {
  const remotes = [...entries].filter(([, entry]) => !entry.process && entry.remoteBaseUrl)
  const dropped: string[] = []

  await Promise.all(
    remotes.map(async ([profile, entry]) => {
      const baseUrl = String(entry.remoteBaseUrl).replace(/\/+$/, '')

      try {
        if (!entry.connectionPromise) {
          throw new Error('Remote backend descriptor is unavailable.')
        }

        const connection = await entry.connectionPromise
        await probe(connection, '/api/status', { timeoutMs: REMOTE_LIVENESS_TIMEOUT_MS })
        tracker.recordSuccess(baseUrl)
      } catch {
        const failure = tracker.recordFailure(baseUrl)

        if (!failure.shouldReset) {
          log(
            `Pooled remote backend for profile "${profile}" failed liveness probe (${failure.failures}/${REMOTE_LIVENESS_FAILURE_LIMIT}); keeping descriptor for retry.`
          )

          return
        }

        log(`Pooled remote backend for profile "${profile}" failed liveness probe; dropping stale descriptor.`)
        stopBackend(profile)
        dropped.push(profile)
      }
    })
  )

  return { dropped }
}

export interface RevalidateSuspectPooledRemoteBackendsOptions<TConnection extends RemoteConnectionDescriptor> {
  entries: Iterable<[string, PooledRemoteEntry<TConnection>]>
  log: (message: string) => void
  probe: (connection: TConnection, path: string, options: { timeoutMs: number }) => Promise<unknown>
  /** Re-dial a retired pool key so the tunnel is rebuilt eagerly, not on the next click. */
  rebuild: (poolKey: string) => Promise<unknown>
  /** Tear down the dead descriptor (pool entry + SSH tunnel/master) for this key. */
  retire: (poolKey: string) => Promise<void> | void
  tracker: RemoteLivenessTracker
}

/**
 * Post-resume sweep of pooled REMOTE descriptors (#93910).
 *
 * After a sleep/wake or network restore every pooled SSH tunnel is suspect:
 * the SSH master died with the network, but the local forward's descriptor is
 * still cached and the renderer keepalive keeps the idle reaper off it. Unlike
 * the background policy in revalidatePooledRemoteBackends — which tolerates a
 * failure streak because transient blips are common in steady state — a
 * suspect descriptor that fails ONE bounded probe after resume is dead: retire
 * it immediately and rebuild, instead of serving "Gateway offline" through two
 * more failure rounds.
 *
 * Bounded by construction: one probe per remote entry per invocation, retire
 * and rebuild each awaited once; the caller coalesces invocations and applies
 * the resume holdoff, so there is no polling loop here. A failed retire skips
 * the rebuild (never dial on top of a descriptor that is still installed) and
 * a failed rebuild is logged and left for the renderer's normal reconnect
 * path — fail closed, never throw out of the sweep.
 */
export async function revalidateSuspectPooledRemoteBackends<TConnection extends RemoteConnectionDescriptor>({
  entries,
  log,
  probe,
  rebuild,
  retire,
  tracker
}: RevalidateSuspectPooledRemoteBackendsOptions<TConnection>): Promise<{ rebuilt: string[]; retired: string[] }> {
  const remotes = [...entries].filter(([, entry]) => !entry.process && entry.remoteBaseUrl)
  const rebuilt: string[] = []
  const retired: string[] = []

  await Promise.all(
    remotes.map(async ([poolKey, entry]) => {
      const baseUrl = String(entry.remoteBaseUrl).replace(/\/+$/, '')

      try {
        if (!entry.connectionPromise) {
          throw new Error('Remote backend descriptor is unavailable.')
        }

        const connection = await entry.connectionPromise
        await probe(connection, '/api/status', { timeoutMs: REMOTE_LIVENESS_TIMEOUT_MS })
        tracker.recordSuccess(baseUrl)

        return
      } catch (probeError) {
        log(
          `Pooled remote backend "${poolKey}" failed its post-resume probe (${probeError instanceof Error ? probeError.message : String(probeError)}); rebuilding tunnel.`
        )
      }

      try {
        await retire(poolKey)
      } catch (retireError) {
        // The dead entry may still be installed; rebuilding on top of it could
        // double-dial one scope. Leave it — the dispatch-time probe retires it
        // on the next use.
        log(
          `Pooled remote backend "${poolKey}" could not be retired after resume (${retireError instanceof Error ? retireError.message : String(retireError)}); leaving descriptor for dispatch-time recovery.`
        )

        return
      }

      retired.push(poolKey)
      // The rebuilt tunnel must start from a clean failure state; stale
      // pre-sleep failures should not count against the fresh descriptor.
      tracker.recordSuccess(baseUrl)

      try {
        await rebuild(poolKey)
        rebuilt.push(poolKey)
      } catch (rebuildError) {
        log(
          `Pooled remote backend "${poolKey}" could not be rebuilt after resume (${rebuildError instanceof Error ? rebuildError.message : String(rebuildError)}); renderer reconnect will retry.`
        )
      }
    })
  )

  return { rebuilt, retired }
}

// macOS fires 'resume' and 'unlock-screen' near-simultaneously on wake, and a
// flapping Wi-Fi association can restore the network several times in a few
// seconds. One sweep per window is enough: the sweep itself probes every
// remote entry, and the renderer's revalidate IPC covers anything that dies
// later. Keep this comfortably above the dispatch probe timeout so overlapping
// signals can never queue back-to-back sweeps into a hot loop.
export const POWER_RESUME_REVALIDATION_HOLDOFF_MS = 15_000

export interface AttachPowerResumeRemoteRevalidationOptions {
  log: (message: string) => void
  now?: () => number
  // Method syntax (bivariant) so Electron's overloaded PowerMonitor.on
  // satisfies this structural seam while tests can pass a tiny fake.
  powerMonitor: { on(event: 'resume' | 'unlock-screen', listener: () => void): unknown }
  revalidate: () => Promise<unknown>
}

/**
 * Wire the suspect-pool sweep to the Electron powerMonitor seam (#93910).
 *
 * Returns the trigger so tests (and the network-restore nudge, if main ever
 * wants one) can drive the exact code path the events run. The trigger is a
 * plain function: holdoff first (one sweep per wake window, never a hot
 * loop), then a fire-and-forget revalidation whose rejection is logged and
 * swallowed — a broken sweep must never take down the resume handler or wedge
 * future wakes.
 */
export function attachPowerResumeRemoteRevalidation({
  log,
  now = Date.now,
  powerMonitor,
  revalidate
}: AttachPowerResumeRemoteRevalidationOptions): () => Promise<void> {
  let lastKickAt: null | number = null

  const trigger = async (): Promise<void> => {
    const at = now()

    if (lastKickAt !== null && at - lastKickAt < POWER_RESUME_REVALIDATION_HOLDOFF_MS) {
      return
    }

    lastKickAt = at

    try {
      await revalidate()
    } catch (error) {
      log(
        `Post-resume remote revalidation failed (${error instanceof Error ? error.message : String(error)}); will retry on the next wake or renderer reconnect.`
      )
    }
  }

  powerMonitor.on('resume', () => void trigger())
  powerMonitor.on('unlock-screen', () => void trigger())

  return trigger
}

/**
 * Probe the cached primary remote connection and apply the failure policy.
 * The caller owns single-flight coordination; identity checks here ensure an
 * old async result cannot mutate or reset a replacement connection.
 */
export async function revalidateRemoteConnection<TConnection extends RemoteConnectionDescriptor>({
  confirmDrop,
  connectionPromise,
  currentConnectionPromise,
  log,
  probe,
  resetConnection,
  tracker
}: RevalidateRemoteConnectionOptions<TConnection>): Promise<RemoteRevalidationResult> {
  let connection: TConnection

  try {
    connection = await connectionPromise
  } catch {
    // The cached boot already rejected; its own recovery path will clear it.
    return { ok: true, rebuilt: false }
  }

  if (currentConnectionPromise() !== connectionPromise) {
    return { ok: true, rebuilt: false }
  }

  if (connection.mode !== 'remote' || !connection.baseUrl) {
    return { ok: true, rebuilt: false }
  }

  const baseUrl = connection.baseUrl.replace(/\/+$/, '')

  try {
    await probe(connection, '/api/status', { timeoutMs: REMOTE_LIVENESS_TIMEOUT_MS })

    if (currentConnectionPromise() !== connectionPromise) {
      return { ok: true, rebuilt: false }
    }

    tracker.recordSuccess(baseUrl)

    return { ok: true, rebuilt: false }
  } catch {
    if (currentConnectionPromise() !== connectionPromise) {
      return { ok: true, rebuilt: false }
    }

    const failure = tracker.recordFailure(baseUrl)

    if (!failure.shouldReset) {
      log(
        `Cached remote Hermes backend failed liveness probe (${failure.failures}/${REMOTE_LIVENESS_FAILURE_LIMIT}); keeping connection for retry.`
      )

      return { ok: true, rebuilt: false }
    }

    if (confirmDrop) {
      log(
        `Cached remote Hermes backend hit its liveness failure limit; confirming with a transport check and one re-probe after ${HOST_EVENT_BACKOFF_MS}ms before dropping.`
      )
      await confirmDrop.backoff()

      if (currentConnectionPromise() !== connectionPromise) {
        return { ok: true, rebuilt: false }
      }

      if (await cachedRemoteConnectionSurvivedBackoff(connection, confirmDrop, probe)) {
        // recordFailure already consumed the streak when it tripped the limit,
        // so the kept connection restarts from a clean slate.
        tracker.recordSuccess(baseUrl)
        log('Cached remote Hermes backend answered after the confirmation backoff; keeping the connection.')

        return { ok: true, rebuilt: false }
      }
    }

    log('Cached remote Hermes backend failed liveness probe; dropping stale connection.')
    resetConnection()

    return { ok: true, rebuilt: true }
  }
}

/** Transport first (cheap), then one real re-probe. Either failure drops. */
async function cachedRemoteConnectionSurvivedBackoff<TConnection extends RemoteConnectionDescriptor>(
  connection: TConnection,
  confirmDrop: RemoteDropConfirmation,
  probe: (connection: TConnection, path: string, options: { timeoutMs: number }) => Promise<unknown>
): Promise<boolean> {
  try {
    if (confirmDrop.transportAlive && !(await confirmDrop.transportAlive())) {
      return false
    }

    await probe(connection, '/api/status', { timeoutMs: REMOTE_LIVENESS_TIMEOUT_MS })

    return true
  } catch {
    return false
  }
}
