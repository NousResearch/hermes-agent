/**
 * Single flight + exponential backoff for SILENT OAuth login windows.
 *
 * A gated remote gateway answers every request that arrives without a session
 * cookie with the auth gate's `401 {reason: "no_cookie"}` refusal, and each
 * refused request independently answers that with ONE silent re-login plus a
 * resubmission (`retryCookie401WithLogin`). A burst of them — the dashboard
 * SPA's boot fan-out, the sidebar polls, the roster — therefore opened one
 * BrowserWindow PER REQUEST, and every window loaded the same gated SPA and
 * issued the same fan-out again. Measured on a real deployment (2026-10-05,
 * desktop.log + the gateway's dashboard-auth.log): 188 refused requests and
 * ~100 login windows in one hour while the identity provider was unreachable
 * for ~8 minutes behind them.
 *
 * One silent login per gateway is the correct unit: the jar either ends up with
 * session cookies or it does not, and N windows racing for it cannot beat one.
 * This holds the single flight for its duration and, after a failure, backs the
 * next attempt off, so a login that cannot complete (identity provider down,
 * portal unreachable, chooser waiting for a human) is not re-attempted once per
 * incoming request.
 *
 * `silent` also selects the URL the window loads, which is why only silent
 * attempts are coalesced here:
 *   - silent=false (default): load ``/login`` — the public interstitial that
 *     renders the "Log in with X" provider chooser. This is the interactive
 *     remote-gateway login the settings UI drives, and it is the user's own
 *     gesture: it always gets its own window, immediately, never held by a
 *     cooldown it did not cause.
 *   - silent=true: load the PROTECTED root ``/`` instead. ``/login`` is a public
 *     route, so loading it NEVER triggers the gate's auto-SSO and always shows
 *     the chooser. Loading a protected page with no session cookie makes the
 *     gate run ``_auto_sso_response``: single registered provider + a live
 *     portal session in this partition → a silent 302 through
 *     ``/auth/login`` → portal ``/oauth/authorize`` (auto-approves org members)
 *     → ``/auth/callback``, which sets the gateway cookie with NO interactive
 *     prompt. This is the per-agent cloud cascade (decisions.md Q5).
 */

export interface SilentLoginCoalescerOptions {
  /** Injectable clock (ms). */
  now?: () => number
  /** First failure's cooldown; doubles per consecutive failure. */
  baseDelayMs?: number
  /** Ceiling for the backoff, so a permanently broken login still retries. */
  maxDelayMs?: number
}

/** Raised instead of opening a window while a recent silent login is backing off. */
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

/**
 * Coalescing key: the gateway ORIGIN, because that is the scope one silent login
 * serves — the session cookie belongs to the origin, and silent callers pass no
 * draft connection id, so they resolve to that origin's own jar.
 */
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
   * Run one login window for `baseUrl`. Interactive calls (`silent` unset) are
   * never coalesced or suppressed — they are the user's own gesture. Silent
   * calls share one attempt per gateway and obey the backoff after a failure.
   */
  runFor<T>(baseUrl: string, options: { silent?: boolean }, open: () => Promise<T>): Promise<T> {
    if (options.silent !== true) {
      return open()
    }

    return this.run(loginKeyFor(baseUrl), open)
  }

  /** Milliseconds still to wait before `key` may open another silent login (0 = now). */
  cooldownRemainingMs(key: string): number {
    const failure = this.failures.get(key)

    if (!failure) {
      return 0
    }

    return Math.max(0, failure.blockedUntil - this.now())
  }

  /**
   * Run `login` for `key`, sharing one attempt between concurrent callers and
   * refusing to start another while the previous failure is still backing off.
   */
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
      // Started synchronously: a burst arriving in one tick has to share this
      // attempt, which is exactly the shape a fan-out of refused requests makes.
      started = login()
    } catch (error) {
      this.recordFailure(key)

      return Promise.reject(error)
    }

    const attempt: Promise<T> = started.then(
      value => {
        // A live session clears the history: the next genuine expiry logs in
        // immediately rather than inheriting a stale cooldown.
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
