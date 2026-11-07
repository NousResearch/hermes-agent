/**
 * One OAuth login window per gateway cookie jar.
 *
 * A gated remote gateway answers every cookie-less request with
 * `401 {reason: "no_cookie"}`, and each refused request used to answer that
 * with its own login window (`retryCookie401WithLogin` → `openOauthLoginWindow`).
 * After a session lapse (overnight sleep, a gateway restart, an IdP outage)
 * every background poll opened one, and the user woke to a stack of them.
 *
 * Rules, per (partition, origin), the scope one cookie jar serves:
 *   - At most one login flow is in flight. A second start would also replace
 *     the gateway's single PKCE state cookie and fail the first callback with
 *     `state_mismatch`.
 *   - Background work (no user gesture) joins a running flow but waits at most
 *     `joinTimeoutMs`, so a window the user left open cannot wedge every
 *     request for that gateway.
 *   - A failed attempt holds the next background attempt back with
 *     exponential backoff. The hold is time-based, not "until the user
 *     clicks": the first attempt after a wake fails while Wi-Fi/VPN reconnect,
 *     and the next one recovers on its own.
 *   - The hold is bounded in time AND in count: after `maxConsecutiveFailures`
 *     consecutive failed attempts a background caller is refused outright
 *     (`SilentLoginExhaustedError`) and no window is opened at all. A gateway
 *     whose credential is genuinely dead (rotated secret, revoked session)
 *     must not be hammered forever at the backoff ceiling — the exhaustion is
 *     reported once (`onExhausted`) so the UI can present "reconnection
 *     required" with a manual action as the only way forward.
 *   - A foreground caller (the user's click) is never held by a cooldown or by
 *     an exhaustion: it IS the manual action. If a hidden attempt is already
 *     running it is revealed instead of opening a competing window, and a
 *     repeat click focuses the open one. The first successful login for the
 *     key clears the failure ledger (`onLoginRecovered` reports the episode
 *     closing).
 */

export type LoginIntent = 'background' | 'foreground'

/** A started login: its outcome, plus a way to put its window in front of the user. */
export interface LoginAttempt<T> {
  result: Promise<T>
  reveal: () => void
}

export interface OauthLoginFlightsOptions {
  now?: () => number
  baseDelayMs?: number
  maxDelayMs?: number
  joinTimeoutMs?: number
  /** Consecutive failed attempts after which background logins stop entirely (and stay stopped until a manual login succeeds). */
  maxConsecutiveFailures?: number
  /** Fired once when a key crosses into exhaustion — the UI's "reconnection required" signal. */
  onExhausted?: (key: string) => void
  /** Fired when a successful login closes a failure episode for the key. */
  onLoginRecovered?: (key: string) => void
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

export class SilentLoginExhaustedError extends Error {
  readonly consecutiveFailures: number

  constructor(consecutiveFailures: number) {
    super(
      `Automatic sign-in for this gateway failed ${consecutiveFailures} times in a row; no further attempts will open ` +
        'until you sign in manually.'
    )
    this.name = 'SilentLoginExhaustedError'
    this.consecutiveFailures = consecutiveFailures
  }
}

export class LoginJoinTimeoutError extends Error {
  constructor(timeoutMs: number) {
    super(`A sign-in for this gateway is still open; stopped waiting after ${Math.ceil(timeoutMs / 1000)}s.`)
    this.name = 'LoginJoinTimeoutError'
  }
}

/** Cookies live per jar and per origin, and so does a login flow. */
export function loginFlightKey(partition: string, baseUrl: string): string {
  let origin = baseUrl

  try {
    origin = new URL(baseUrl).origin
  } catch {
    // an unparsable URL still keys consistently
  }

  return `${partition} ${origin}`
}

/** The inverse of `loginFlightKey`: the origin half of a key, for reporting. */
export function originFromFlightKey(key: string): string {
  const origin = key.slice(key.indexOf(' ') + 1)

  return origin || key
}

interface Flight {
  result: Promise<unknown>
  reveal: () => void
}

export class OauthLoginFlights {
  private readonly now: () => number
  private readonly baseDelayMs: number
  private readonly maxDelayMs: number
  private readonly joinTimeoutMs: number
  private readonly maxConsecutiveFailures: number
  private readonly onExhausted: (key: string) => void
  private readonly onLoginRecovered: (key: string) => void
  private readonly inFlight = new Map<string, Flight>()
  private readonly failures = new Map<string, { consecutive: number; blockedUntil: number }>()

  constructor(options: OauthLoginFlightsOptions = {}) {
    this.now = options.now ?? Date.now
    this.baseDelayMs = options.baseDelayMs ?? 5_000
    // A permanently dead credential must not be probed every minute either:
    // the ceiling is 5 minutes, and `maxConsecutiveFailures` ends the series
    // outright long before a capped delay can repeat many times.
    this.maxDelayMs = options.maxDelayMs ?? 300_000
    this.joinTimeoutMs = options.joinTimeoutMs ?? 15_000
    this.maxConsecutiveFailures = Math.max(1, options.maxConsecutiveFailures ?? 3)
    this.onExhausted = options.onExhausted ?? (() => undefined)
    this.onLoginRecovered = options.onLoginRecovered ?? (() => undefined)
  }

  isRunning(key: string): boolean {
    return this.inFlight.has(key)
  }

  cooldownRemainingMs(key: string): number {
    const failure = this.failures.get(key)

    return failure ? Math.max(0, failure.blockedUntil - this.now()) : 0
  }

  /** `start` runs only when `key` has no flight; joiners never open a window. */
  run<T>(key: string, intent: LoginIntent, start: () => LoginAttempt<T>): Promise<T> {
    const running = this.inFlight.get(key)

    if (running) {
      if (intent === 'foreground') {
        running.reveal()

        return running.result as Promise<T>
      }

      return this.boundedJoin(running.result as Promise<T>)
    }

    const remaining = intent === 'background' ? this.cooldownRemainingMs(key) : 0

    if (remaining > 0) {
      return Promise.reject(new SilentLoginSuppressedError(remaining))
    }

    const failure = this.failures.get(key)

    if (intent === 'background' && (failure?.consecutive ?? 0) >= this.maxConsecutiveFailures) {
      return Promise.reject(new SilentLoginExhaustedError(failure?.consecutive ?? 0))
    }

    let attempt: LoginAttempt<T>

    try {
      attempt = start()
    } catch (error) {
      this.recordFailure(key)

      return Promise.reject(error)
    }

    const hadFailures = failure !== undefined

    const result: Promise<T> = attempt.result.then(
      value => {
        if (hadFailures) {
          this.onLoginRecovered(key)
        }

        this.failures.delete(key)

        return value
      },
      error => {
        this.recordFailure(key)

        throw error
      }
    )

    const flight: Flight = { result, reveal: attempt.reveal }
    this.inFlight.set(key, flight)

    const release = (): void => {
      if (this.inFlight.get(key) === flight) {
        this.inFlight.delete(key)
      }
    }

    void result.then(release, release)

    return result
  }

  private boundedJoin<T>(result: Promise<T>): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      const timer = setTimeout(() => reject(new LoginJoinTimeoutError(this.joinTimeoutMs)), this.joinTimeoutMs)

      result.then(
        value => {
          clearTimeout(timer)
          resolve(value)
        },
        error => {
          clearTimeout(timer)
          reject(error)
        }
      )
    })
  }

  private recordFailure(key: string): void {
    const consecutive = (this.failures.get(key)?.consecutive ?? 0) + 1
    const delay = Math.min(this.baseDelayMs * 2 ** (consecutive - 1), this.maxDelayMs)

    this.failures.set(key, { consecutive, blockedUntil: this.now() + delay })

    if (consecutive === this.maxConsecutiveFailures) {
      this.onExhausted(key)
    }
  }
}
