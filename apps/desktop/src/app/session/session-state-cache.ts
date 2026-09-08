import type { ClientSessionState } from '../types'

export const DEFAULT_WARM_SESSION_TRANSCRIPT_COUNT = 24
export const DEFAULT_WARM_SESSION_TRANSCRIPT_BYTES = 32 * 1024 * 1024

/**
 * A busy/awaiting entry whose state has not changed for this long is an
 * orphaned mid-turn session (#95276): while the authoritative store still
 * claims work for its runtime id, its frozen flags fail #isWarmSettled
 * forever, pinning its transcript outside both LRU bounds. Chosen far beyond
 * the five-minute session watchdog so legitimate quiet stretches (long tool
 * runs stream no events) never trip it, while still bounding how long a dead
 * turn can hold memory.
 */
export const DEFAULT_STALLED_SESSION_MS = 30 * 60 * 1000

interface SessionStateCacheLimits {
  maxBytes?: number
  maxCount?: number
  /** Silence past which a busy/awaiting entry becomes an eviction candidate. */
  stalledMs?: number
  /** Injectable clock for deterministic tests. */
  now?: () => number
}

interface SessionStateCacheCallbacks {
  isReferenced: (runtimeId: string, state: ClientSessionState) => boolean
  onEvict: (runtimeId: string, state: ClientSessionState) => void
  /** Optional liveness check for a cached snapshot's in-flight claims. A
   *  connection death mid-turn orphans snapshots: the respawned backend
   *  re-mints runtime ids, so their frozen busy/awaitingResponse flags never
   *  receive a settling publish (#95189) and would pin megabytes of warm
   *  transcript per reconnect cycle until restart. When wired, those flags
   *  only block eviction while the authoritative store still claims work for
   *  the same runtime id; without the probe they always block. */
  isAuthoritativelyActive?: (runtimeId: string, state: ClientSessionState) => boolean
}

function transcriptBytes(state: ClientSessionState): number {
  if (state.messages.length === 0) {
    return 0
  }

  // JS strings occupy two bytes per UTF-16 code unit. JSON also accounts for
  // ids, part tags, tool payloads, attachment metadata, and error text without
  // retaining a second serialized copy in the cache.
  return JSON.stringify(state.messages).length * 2
}

function hasDraftOrInFlightMessage(state: ClientSessionState): boolean {
  return state.messages.some(message => message.pending === true)
}

/**
 * Runtime state map whose settled, unreferenced transcripts form a weighted
 * LRU. Live/visible states and unsaved drafts are outside both limits.
 */
export class SessionStateCache extends Map<string, ClientSessionState> {
  readonly #callbacks: SessionStateCacheCallbacks
  readonly #maxBytes: number
  readonly #maxCount: number
  readonly #stalledMs: number
  readonly #now: () => number
  readonly #recency = new Map<string, number>()
  // Last time each entry's state was replaced via set(). updateSessionState
  // skips writes for unchanged states, so this is "last event activity" —
  // reads must not count, or polling would keep dead sessions fresh forever.
  readonly #mutations = new Map<string, number>()
  #clock = 0

  constructor(callbacks: SessionStateCacheCallbacks, limits: SessionStateCacheLimits = {}) {
    super()
    this.#callbacks = callbacks
    this.#maxBytes = limits.maxBytes ?? DEFAULT_WARM_SESSION_TRANSCRIPT_BYTES
    this.#maxCount = limits.maxCount ?? DEFAULT_WARM_SESSION_TRANSCRIPT_COUNT
    this.#stalledMs = limits.stalledMs ?? DEFAULT_STALLED_SESSION_MS
    this.#now = limits.now ?? (() => Date.now())
  }

  override get(runtimeId: string): ClientSessionState | undefined {
    const state = super.get(runtimeId)

    if (state) {
      this.#touch(runtimeId)
    }

    return state
  }

  override set(runtimeId: string, state: ClientSessionState): this {
    super.set(runtimeId, state)
    this.#touch(runtimeId)
    this.#mutations.set(runtimeId, this.#now())

    return this
  }

  override delete(runtimeId: string): boolean {
    this.#recency.delete(runtimeId)
    this.#mutations.delete(runtimeId)

    return super.delete(runtimeId)
  }

  override clear(): void {
    this.#recency.clear()
    this.#mutations.clear()
    super.clear()
  }

  prune(): void {
    this.#evictStalled()

    const candidates: Array<{ bytes: number; runtimeId: string; state: ClientSessionState; touched: number }> = []
    let bytes = 0

    for (const [runtimeId, state] of this.entries()) {
      if (!this.#isWarmSettled(runtimeId, state)) {
        continue
      }

      const weight = transcriptBytes(state)
      candidates.push({ bytes: weight, runtimeId, state, touched: this.#recency.get(runtimeId) ?? 0 })
      bytes += weight
    }

    let count = candidates.length

    if (count <= this.#maxCount && bytes <= this.#maxBytes) {
      return
    }

    candidates.sort((a, b) => a.touched - b.touched)

    for (const candidate of candidates) {
      if (count <= this.#maxCount && bytes <= this.#maxBytes) {
        break
      }

      // References and activity can change between insertion and pruning.
      const current = super.get(candidate.runtimeId)

      if (current !== candidate.state || !this.#isWarmSettled(candidate.runtimeId, current)) {
        continue
      }

      super.delete(candidate.runtimeId)
      this.#recency.delete(candidate.runtimeId)
      count -= 1
      bytes -= candidate.bytes
      this.#callbacks.onEvict(candidate.runtimeId, candidate.state)
    }
  }

  /**
   * Orphaned mid-turn entries never satisfy #isWarmSettled, so they are
   * invisible to the weighted LRU and would pin their transcripts forever.
   * Unlike that sweep this one is unconditional — it exists to bound memory,
   * not to serve it — but it keeps every protection that makes eviction safe:
   * persisted sessions only, no drafts or in-flight messages, waits that are
   * on the user rather than on a dead turn, and live references win.
   */
  #evictStalled(): void {
    const stalled: Array<{ runtimeId: string; state: ClientSessionState }> = []

    for (const [runtimeId, state] of this.entries()) {
      if (this.#isStalled(runtimeId, state)) {
        stalled.push({ runtimeId, state })
      }
    }

    for (const candidate of stalled) {
      // References and activity can change between detection and eviction.
      const current = super.get(candidate.runtimeId)

      if (current !== candidate.state || !this.#isStalled(candidate.runtimeId, current)) {
        continue
      }

      super.delete(candidate.runtimeId)
      this.#recency.delete(candidate.runtimeId)
      this.#mutations.delete(candidate.runtimeId)
      this.#callbacks.onEvict(candidate.runtimeId, candidate.state)
    }
  }

  #isStalled(runtimeId: string, state: ClientSessionState): boolean {
    const mutatedAt = this.#mutations.get(runtimeId)

    return (
      Boolean(state.storedSessionId) &&
      (state.busy || state.awaitingResponse) &&
      // A blocking prompt is waiting on a human, not on a dead turn.
      !state.needsInput &&
      !hasDraftOrInFlightMessage(state) &&
      mutatedAt !== undefined &&
      this.#now() - mutatedAt >= this.#stalledMs &&
      !this.#callbacks.isReferenced(runtimeId, state)
    )
  }

  #isWarmSettled(runtimeId: string, state: ClientSessionState): boolean {
    // In-flight claims pin a transcript only while they are trustworthy: with
    // an authority probe wired, a frozen busy/awaitingResponse on an orphaned
    // snapshot stops blocking eviction (see callback docs). Without one, the
    // legacy behavior holds and the flags always block.
    if (
      (state.busy || state.awaitingResponse) &&
      this.#callbacks.isAuthoritativelyActive?.(runtimeId, state) !== false
    ) {
      return false
    }

    return (
      Boolean(state.storedSessionId) &&
      state.messages.length > 0 &&
      !state.needsInput &&
      !hasDraftOrInFlightMessage(state) &&
      !this.#callbacks.isReferenced(runtimeId, state)
    )
  }

  #touch(runtimeId: string): void {
    this.#clock += 1
    this.#recency.set(runtimeId, this.#clock)
  }
}
