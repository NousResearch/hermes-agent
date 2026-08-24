import type { ClientSessionState } from '../types'

export const DEFAULT_WARM_SESSION_TRANSCRIPT_COUNT = 24
export const DEFAULT_WARM_SESSION_TRANSCRIPT_BYTES = 32 * 1024 * 1024

interface SessionStateCacheLimits {
  maxBytes?: number
  maxCount?: number
}

interface SessionStateCacheCallbacks {
  isReferenced: (runtimeId: string, state: ClientSessionState) => boolean
  onEvict: (runtimeId: string, state: ClientSessionState) => void
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
  readonly #recency = new Map<string, number>()
  #clock = 0

  constructor(callbacks: SessionStateCacheCallbacks, limits: SessionStateCacheLimits = {}) {
    super()
    this.#callbacks = callbacks
    this.#maxBytes = limits.maxBytes ?? DEFAULT_WARM_SESSION_TRANSCRIPT_BYTES
    this.#maxCount = limits.maxCount ?? DEFAULT_WARM_SESSION_TRANSCRIPT_COUNT
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

    return this
  }

  override delete(runtimeId: string): boolean {
    this.#recency.delete(runtimeId)

    return super.delete(runtimeId)
  }

  override clear(): void {
    this.#recency.clear()
    super.clear()
  }

  prune(): void {
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

  #isWarmSettled(runtimeId: string, state: ClientSessionState): boolean {
    return (
      Boolean(state.storedSessionId) &&
      state.messages.length > 0 &&
      !state.busy &&
      !state.awaitingResponse &&
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
