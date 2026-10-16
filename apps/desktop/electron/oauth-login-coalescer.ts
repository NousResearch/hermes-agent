/**
 * One silent OAuth login window per gateway origin.
 *
 * A gated remote gateway answers every cookie-less request with
 * `401 {reason: "no_cookie"}`, and each refused request answers that with one
 * silent re-login (`retryCookie401WithLogin` → `openOauthLoginWindow`). A burst
 * — boot, sidebar polls, a wake while the network is still down — therefore
 * opened one BrowserWindow per request.
 *
 * Silent callers share one attempt: the jar either gains a session cookie or it
 * does not. After a failure the next silent attempt waits, so a closed or
 * unreachable window is not replaced once per incoming request. An interactive
 * sign-in is the user's own click and always opens immediately.
 *
 * `silent` also selects the URL the window loads:
 *   - silent=false: `/login`, the provider chooser.
 *   - silent=true: the protected root `/`, so the gate can auto-SSO through
 *     portal `/oauth/authorize` back to `/auth/callback` without a chooser.
 */

export interface SilentLoginCoalescerOptions {
  now?: () => number
  baseDelayMs?: number
  maxDelayMs?: number
}

export class SilentLoginSuppressedError extends Error {
  readonly retryAfterMs: number

  constructor(retryAfterMs: number) {
    super(
      `A silent sign-in for this gateway failed recently; the next attempt is held for ${Math.ceil(retryAfterMs / 1000)}s.`
    )
    this.name = 'SilentLoginSuppressedError'
    this.retryAfterMs = retryAfterMs
  }
}

/** Session cookies belong to the origin, which is the scope one silent login serves. */
export function loginKeyFor(baseUrl: string): string {
  try {
    return new URL(baseUrl).origin
  } catch {
    return baseUrl
  }
}

export class SilentLoginCoalescer {
  private readonly now: () => number
  private readonly baseDelayMs: number
  private readonly maxDelayMs: number
  private readonly inFlight = new Map<string, Promise<unknown>>()
  private readonly failures = new Map<string, { consecutive: number; blockedUntil: number }>()

  constructor(options: SilentLoginCoalescerOptions = {}) {
    this.now = options.now ?? Date.now
    this.baseDelayMs = options.baseDelayMs ?? 5_000
    this.maxDelayMs = options.maxDelayMs ?? 60_000
  }

  /**
   * Open one login for `baseUrl`. The opener is always called as
   * `open(baseUrl, options)` — a zero-arg callback drops the gateway URL and
   * no window can load.
   */
  runFor<T, O extends { silent?: boolean }>(
    baseUrl: string,
    options: O,
    open: (baseUrl: string, options: O) => Promise<T>
  ): Promise<T> {
    if (options?.silent !== true) {
      return open(baseUrl, options)
    }

    return this.run(loginKeyFor(baseUrl), () => open(baseUrl, options))
  }

  cooldownRemainingMs(key: string): number {
    const failure = this.failures.get(key)

    if (!failure) {
      return 0
    }

    return Math.max(0, failure.blockedUntil - this.now())
  }

  run<T>(key: string, login: () => Promise<T>): Promise<T> {
    const running = this.inFlight.get(key)

    if (running) {
      return running as Promise<T>
    }

    const remaining = this.cooldownRemainingMs(key)

    if (remaining > 0) {
      return Promise.reject(new SilentLoginSuppressedError(remaining))
    }

    let started: Promise<T>

    try {
      started = login()
    } catch (error) {
      this.recordFailure(key)

      return Promise.reject(error)
    }

    const attempt: Promise<T> = started.then(
      value => {
        this.failures.delete(key)

        return value
      },
      error => {
        this.recordFailure(key)

        throw error
      }
    )

    this.inFlight.set(key, attempt)

    const release = (): void => {
      if (this.inFlight.get(key) === attempt) {
        this.inFlight.delete(key)
      }
    }

    void attempt.then(release, release)

    return attempt
  }

  private recordFailure(key: string): void {
    const consecutive = (this.failures.get(key)?.consecutive ?? 0) + 1
    const delay = Math.min(this.baseDelayMs * 2 ** (consecutive - 1), this.maxDelayMs)

    this.failures.set(key, { consecutive, blockedUntil: this.now() + delay })
  }
}
