/**
 * backend-dial-claim.ts
 *
 * Single-owner reconnect/dial claim for backend spawns, keyed by the pool
 * scope key from backendScopeKey(connectionId, profile) (#90812).
 *
 * Why this exists: reconnectGateway()'s in-flight lock lives at renderer
 * module scope, so it only dedupes reconnects INSIDE one window. Two windows
 * (main + a session pop-out) racing the same wake both invoke the main-process
 * dial IPC, and for a pooled SSH connection the loser of the pool-entry race
 * could bootstrap a duplicate remote backend. Electron main is the single
 * owner of backend lifecycles, so the claim belongs here: the first dial for a
 * (connectionId, profile) key runs; every concurrent caller for the same key
 * awaits and receives that first dial's result.
 *
 * Bounded by construction: a claim exists only while its dial promise is
 * unsettled — both outcomes release it, so a failed dial is never cached and
 * the next reconnect attempt runs fresh (fail closed, not latched).
 */
/** What the callers of a forgotten dial are told: retry, the profile has another route now. */
export const DIAL_ROUTE_REPLACED = 'The route this backend dial was waiting on was replaced.'

interface DialClaim {
  promise: Promise<unknown>
  replaced: (error: Error) => void
}

export class BackendDialClaims {
  readonly #inflightByKey = new Map<string, DialClaim>()

  /** Whether a dial for this key is currently in flight (test/diagnostic seam). */
  inFlight(key: string): boolean {
    return this.#inflightByKey.has(key)
  }

  /**
   * The route this key's dial is waiting on was replaced: the next caller
   * dials the new route, and the callers already waiting are rejected now
   * (DIAL_ROUTE_REPLACED) so they ask again, instead of sitting out a wait
   * that can last the whole readiness limit. The old dial itself runs on
   * unobserved; its settle releases nothing, the claim is no longer its own.
   */
  forget(key: string): void {
    const claim = this.#inflightByKey.get(key)

    if (claim) {
      this.#inflightByKey.delete(key)
      claim.replaced(new Error(DIAL_ROUTE_REPLACED))
    }
  }

  run<T>(key: string, dial: () => Promise<T> | T): Promise<T> {
    const existing = this.#inflightByKey.get(key)

    if (existing) {
      return existing.promise as Promise<T>
    }

    // Start the dial eagerly so the first caller's spawn is already in flight
    // when a concurrent caller arrives; a synchronously-throwing dial is
    // converted into a rejection of THIS claim so it cannot bypass the seam.
    let dialed: Promise<T>

    try {
      dialed = Promise.resolve(dial())
    } catch (error) {
      dialed = Promise.reject(error)
    }

    let replaced: (error: Error) => void = () => undefined

    const pending = Promise.race([
      dialed,
      new Promise<never>((_resolve, reject) => {
        replaced = reject
      })
    ])

    const claim: DialClaim = { promise: pending, replaced }

    const release = () => {
      if (this.#inflightByKey.get(key) === claim) {
        this.#inflightByKey.delete(key)
      }
    }

    this.#inflightByKey.set(key, claim)
    // The claim ends when the dial itself settles, in the same microtask as
    // before forget existed; its callers hear one microtask later (the race).
    void dialed.then(release, release)
    // Release on both outcomes without creating an unhandled rejected branch.
    void pending.then(release, release)

    return pending
  }
}
