export const REMOTE_LIVENESS_TIMEOUT_MS = 10_000
export const REMOTE_LIVENESS_FAILURE_LIMIT = 3

export interface RemoteLivenessFailure {
  failures: number
  shouldReset: boolean
}

/**
 * Tracks consecutive remote liveness failures independently per gateway.
 * A successful probe clears the streak, and reaching the limit consumes it so
 * a rebuilt connection starts from a clean state.
 */
export class RemoteLivenessTracker {
  readonly #failureLimit: number
  readonly #failuresByBaseUrl = new Map<string, number>()

  constructor(failureLimit = REMOTE_LIVENESS_FAILURE_LIMIT) {
    if (!Number.isInteger(failureLimit) || failureLimit < 1) {
      throw new Error('Remote liveness failure limit must be a positive integer.')
    }

    this.#failureLimit = failureLimit
  }

  recordSuccess(baseUrl: string): void {
    this.#failuresByBaseUrl.delete(baseUrl)
  }

  recordFailure(baseUrl: string): RemoteLivenessFailure {
    const failures = (this.#failuresByBaseUrl.get(baseUrl) ?? 0) + 1
    const shouldReset = failures >= this.#failureLimit

    if (shouldReset) {
      this.#failuresByBaseUrl.delete(baseUrl)
    } else {
      this.#failuresByBaseUrl.set(baseUrl, failures)
    }

    return { failures, shouldReset }
  }

  clear(): void {
    this.#failuresByBaseUrl.clear()
  }
}
