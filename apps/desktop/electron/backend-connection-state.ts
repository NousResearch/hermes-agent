export type BackendConnectionAttempt<TConnection> = {
  generation: number
  promise: Promise<TConnection> | null
  /**
   * True when a later caller joined a start that is still dialing (#127974).
   * The owner reads its own value in the same synchronous turn as
   * startAttempt(), before any joiner can run.
   */
  joined: boolean
}

export type BackendProcessOwner<TProcess> = {
  generation: number
  process: TProcess
}

interface PendingBackendStop<TProcess> {
  process: TProcess
  completion: Promise<void>
  failed: boolean
}

interface StartFlight<TConnection> {
  attempt: BackendConnectionAttempt<TConnection>
  beganAt: number
  published: Promise<TConnection> | null
  settled: Promise<void>
  settle: () => void
  released: boolean
}

export interface BackendConnectionStateOptions {
  /** Backoff spacing seam: tests inject it so the restart pacing is deterministic. */
  sleep?: (ms: number) => Promise<void>
  /** Upper bound on waiting for a superseded start to settle before dialing again. */
  settleTimeoutMs?: number
  /** Interval before the first replacement dial; it doubles per consecutive supersede. */
  supersedeBackoffBaseMs?: number
  /** Cap on that interval. */
  supersedeBackoffMaxMs?: number
  /** How long a superseded start stays in the wait set when it never settles. */
  staleFlightRetentionMs?: number
  /**
   * How long a live start may stay unpublished before a later start stops
   * joining it. runHermesStart publishes in the same synchronous turn as its
   * startAttempt(), so an unpublished flight seen by a later caller belongs to
   * an owner that threw in between and can never produce a connection.
   */
  unpublishedStartTimeoutMs?: number
}

export interface BackendConnectionState<TProcess, TConnection> {
  startAttempt(): BackendConnectionAttempt<TConnection>
  setPromise(attempt: BackendConnectionAttempt<TConnection>, promise: Promise<TConnection>): boolean
  isCurrentAttempt(attempt: BackendConnectionAttempt<TConnection>): boolean
  assertCurrentAttempt(attempt: BackendConnectionAttempt<TConnection>): void
  attachProcess(attempt: BackendConnectionAttempt<TConnection>, process: TProcess): BackendProcessOwner<TProcess> | null
  claimProcess(
    attempt: BackendConnectionAttempt<TConnection>,
    process: TProcess,
    claim: (current: TProcess) => Promise<unknown>
  ): Promise<BackendProcessOwner<TProcess> | null>
  clearForCurrentProcess(owner: BackendProcessOwner<TProcess>): boolean
  clearPromiseForAttempt(attempt: BackendConnectionAttempt<TConnection>): boolean
  awaitSupersededStart(): Promise<void>
  getProcess(): TProcess | null
  getPromise(): Promise<TConnection> | null
  getPendingPromise(): Promise<TConnection> | null
  invalidate(): TProcess | null
  stopProcess(stop: (current: TProcess) => Promise<void>): Promise<void>
}

export function createBackendConnectionState<TProcess, TConnection>(
  options: BackendConnectionStateOptions = {}
): BackendConnectionState<TProcess, TConnection> {
  const sleep =
    options.sleep ??
    ((ms: number): Promise<void> =>
      new Promise<void>((resolve: () => void): void => {
        setTimeout(resolve, ms)
      }))
  const settleTimeoutMs = options.settleTimeoutMs ?? 7_000
  const supersedeBackoffBaseMs = options.supersedeBackoffBaseMs ?? 250
  const supersedeBackoffMaxMs = options.supersedeBackoffMaxMs ?? 4_000
  const staleFlightRetentionMs = options.staleFlightRetentionMs ?? 30_000
  const unpublishedStartTimeoutMs = options.unpublishedStartTimeoutMs ?? 1_000

  let generation = 0
  let process: TProcess | null = null
  let promise: Promise<TConnection> | null = null
  let pendingPromise: Promise<TConnection> | null = null
  let stopping: PendingBackendStop<TProcess> | null = null
  let flight: StartFlight<TConnection> | null = null
  let supersededFlights: StartFlight<TConnection>[] = []
  let consecutiveSupersedes = 0

  function createFlight(attempt: BackendConnectionAttempt<TConnection>): StartFlight<TConnection> {
    let release!: () => void

    return {
      attempt,
      beganAt: Date.now(),
      published: null,
      released: false,
      settle: (): void => release(),
      settled: new Promise<void>((resolve: () => void): void => {
        release = resolve
      })
    }
  }

  function releaseFlight(entry: StartFlight<TConnection>, healthy: boolean): void {
    if (entry.released) {
      return
    }

    entry.released = true

    if (flight === entry) {
      flight = null
    }

    supersededFlights = supersededFlights.filter(candidate => candidate !== entry)

    if (healthy) {
      consecutiveSupersedes = 0
    }

    entry.settle()
  }

  // Every deliberate emptying of the live slot is one restart step: the
  // replacement waits the flight out, paced by the backoff below.
  function supersedeLiveFlight(): void {
    if (!flight) {
      return
    }

    consecutiveSupersedes += 1
    supersededFlights.push(flight)
    flight = null
  }

  // ponytail: a live start that never publishes is fenced off by age instead of
  // being cancelled — the owner's later setPromise() is refused because the
  // state no longer owns the flight, so it cannot publish a stowaway connection.
  function reclaimUnpublishedFlight(): void {
    if (!flight || flight.published || Date.now() - flight.beganAt < unpublishedStartTimeoutMs) {
      return
    }

    releaseFlight(flight, false)
  }

  // ponytail: a superseded start that never settles is abandoned after
  // staleFlightRetentionMs instead of being cancelled per-flight — the bounded
  // wait in awaitSupersededStart() is the whole recovery.
  function pruneSupersededFlights(): void {
    const cutoff = Date.now() - staleFlightRetentionMs

    for (const entry of supersededFlights.filter(candidate => candidate.beganAt <= cutoff)) {
      releaseFlight(entry, false)
    }
  }

  function supersedeBackoffMs(consecutive: number): number {
    if (consecutive <= 0) {
      return 0
    }

    return Math.min(supersedeBackoffBaseMs * 2 ** (consecutive - 1), supersedeBackoffMaxMs)
  }

  async function waitForSettlement(pending: Promise<void>[]): Promise<void> {
    if (!pending.length) {
      return
    }

    let timer: ReturnType<typeof setTimeout> | undefined

    try {
      await Promise.race([
        Promise.allSettled(pending),
        new Promise<void>((resolve: () => void): void => {
          timer = setTimeout(resolve, settleTimeoutMs)
        })
      ])
    } finally {
      clearTimeout(timer)
    }
  }

  function invalidate(): TProcess | null {
    const currentProcess = process
    generation += 1
    process = null
    promise = null
    pendingPromise = null
    supersedeLiveFlight()

    return currentProcess
  }

  return {
    startAttempt(): BackendConnectionAttempt<TConnection> {
      if (stopping) {
        throw new Error('The previous backend has not stopped. Retry its shutdown before starting a replacement.')
      }

      // Single-flight (#127974): a reconnect, a renderer retry, or a supervisor
      // respawn landing while a boot is still dialing must join that boot. Minting
      // a second, equally current attempt made both boots spawn; the loser was
      // only discovered once its claim resolved, so it surfaced as
      // `superseded by a newer connection attempt` and its child's exit as a
      // stale backend exit.
      reclaimUnpublishedFlight()

      if (flight) {
        flight.attempt.joined = true

        return flight.attempt
      }

      const attempt: BackendConnectionAttempt<TConnection> = { generation, joined: false, promise: null }

      flight = createFlight(attempt)

      return attempt
    },

    setPromise(attempt: BackendConnectionAttempt<TConnection>, nextPromise: Promise<TConnection>): boolean {
      const owning =
        flight?.attempt === attempt
          ? flight
          : supersededFlights.find(candidate => candidate.attempt === attempt) ?? null

      // A flight this state no longer owns must not publish: its connection
      // belongs to a superseded generation or to a start that never completed.
      if (!owning) {
        return false
      }

      owning.published = nextPromise

      void nextPromise.then(
        (): void => releaseFlight(owning, true),
        (): void => releaseFlight(owning, false)
      )

      if (attempt.generation !== generation) {
        return false
      }

      attempt.promise = nextPromise
      promise = nextPromise
      pendingPromise = nextPromise

      void nextPromise.then(
        () => {
          if (attempt.generation === generation && promise === nextPromise) {
            pendingPromise = null
          }
        },
        () => {
          if (attempt.generation === generation && promise === nextPromise) {
            pendingPromise = null
          }
        }
      )

      return true
    },

    isCurrentAttempt(attempt: BackendConnectionAttempt<TConnection>): boolean {
      return attempt.generation === generation
    },

    assertCurrentAttempt(attempt: BackendConnectionAttempt<TConnection>): void {
      if (attempt.generation !== generation) {
        throw new Error('Hermes backend start was superseded by a newer connection attempt.')
      }
    },

    attachProcess(
      attempt: BackendConnectionAttempt<TConnection>,
      nextProcess: TProcess
    ): BackendProcessOwner<TProcess> | null {
      if (attempt.generation !== generation) {
        return null
      }

      process = nextProcess

      return { generation, process: nextProcess }
    },

    async claimProcess(
      attempt: BackendConnectionAttempt<TConnection>,
      nextProcess: TProcess,
      claim: (current: TProcess) => Promise<unknown>
    ): Promise<BackendProcessOwner<TProcess> | null> {
      const owner = this.attachProcess(attempt, nextProcess)

      if (!owner) {
        return null
      }

      await claim(nextProcess)

      return owner.generation === generation && process === nextProcess ? owner : null
    },

    clearForCurrentProcess(owner: BackendProcessOwner<TProcess>): boolean {
      if (owner.generation !== generation || owner.process !== process) {
        return false
      }

      process = null
      promise = null
      pendingPromise = null

      return true
    },

    clearPromiseForAttempt(attempt: BackendConnectionAttempt<TConnection>): boolean {
      if (attempt.generation !== generation || (promise !== null && attempt.promise !== promise)) {
        return false
      }

      promise = null
      pendingPromise = null

      return true
    },

    /**
     * Bounded, backed-off wait for the starts this generation superseded. A
     * reconnect that invalidated a boot in flight awaits this before dialing its
     * replacement, so the two cannot overlap (#127974) and a restart storm
     * cannot re-dial on top of an unfinished boot.
     */
    async awaitSupersededStart(): Promise<void> {
      pruneSupersededFlights()

      if (!supersededFlights.length) {
        return
      }

      const pending = supersededFlights.map(entry => entry.settled)

      await sleep(supersedeBackoffMs(consecutiveSupersedes))
      await waitForSettlement(pending)
    },

    getProcess(): TProcess | null {
      return process
    },

    getPromise(): Promise<TConnection> | null {
      return promise
    },

    getPendingPromise(): Promise<TConnection> | null {
      return pendingPromise
    },

    invalidate,

    stopProcess(stop: (current: TProcess) => Promise<void>): Promise<void> {
      if (stopping && !stopping.failed) {
        return stopping.completion
      }

      const current = stopping?.process ?? invalidate()

      if (current === null) {
        return Promise.resolve()
      }

      const completion = Promise.resolve()
        .then((): Promise<void> => stop(current))
        .then(
          (): void => {
            stopping = null
          },
          (error: unknown): never => {
            pending.failed = true
            throw error
          }
        )

      const pending: PendingBackendStop<TProcess> = { process: current, completion, failed: false }
      stopping = pending

      return completion
    }
  }
}
