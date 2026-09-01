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

export interface RemoteLivenessFailure {
  failures: number
  shouldReset: boolean
}

interface RemoteConnectionDescriptor {
  baseUrl?: null | string
  mode?: null | string
}

export interface RevalidateRemoteConnectionOptions<TConnection extends RemoteConnectionDescriptor> {
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

interface EnsureHealthyPooledRemoteBackendForDispatchOptions<TConnection extends RemoteConnectionDescriptor> {
  connectionPromise: Promise<TConnection>
  currentConnectionPromise: () => null | Promise<TConnection>
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
 */
export async function ensureHealthyPooledRemoteBackendForDispatch<TConnection extends RemoteConnectionDescriptor>({
  connectionPromise,
  currentConnectionPromise,
  probe,
  reconnect,
  retire
}: EnsureHealthyPooledRemoteBackendForDispatchOptions<TConnection>): Promise<TConnection> {
  let connection: TConnection

  try {
    connection = await connectionPromise

    if (currentConnectionPromise() !== connectionPromise) {
      return reconnect()
    }

    await probe(connection, '/api/status', {
      timeoutMs: POOLED_REMOTE_DISPATCH_PROBE_TIMEOUT_MS
    })
  } catch (error) {
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

export interface RetirePooledRemoteAfterDispatchProbeOptions {
  cancelSshBootstrap: (key: string) => Promise<void> | void
  currentEntry: () => unknown
  error: unknown
  expectedEntry: unknown
  key: string
  log: (message: string) => void
  sourceKind: string
  stopPoolBackend: (key: string) => Promise<void> | void
}

/**
 * Retire one pooled remote descriptor after a failed dispatch probe.
 *
 * Stop the pool entry and cancel in-flight SSH bootstrap for that key, and
 * leave the cached SSH scope state alone. The probe fires on the first miss
 * with a 2.5s budget, so it cannot tell a transient hiccup from a dead host;
 * demolishing the scope there discards a working tunnel and a live remote
 * dashboard that `bootstrapSshConnection` would otherwise reuse. Deciding a
 * connection is actually gone stays with the cached-connection liveness path
 * and its multi-strike policy.
 */
export async function retirePooledRemoteAfterDispatchProbe({
  cancelSshBootstrap,
  currentEntry,
  error,
  expectedEntry,
  key,
  log,
  sourceKind,
  stopPoolBackend
}: RetirePooledRemoteAfterDispatchProbeOptions): Promise<void> {
  // A late failure from an old descriptor must never tear down a newer
  // entry that another caller has already installed.
  if (currentEntry() !== expectedEntry) {
    return
  }

  const detail = error instanceof Error ? error.message : String(error)

  log(`Pooled remote backend "${key}" failed its dispatch probe (${detail}); reconnecting on demand.`)
  await stopPoolBackend(key)

  if (sourceKind === 'ssh') {
    await cancelSshBootstrap(key)
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

    log('Cached remote Hermes backend failed liveness probe; dropping stale connection.')
    resetConnection()

    return { ok: true, rebuilt: true }
  }
}
