/**
 * backend-start-failure.ts
 *
 * Decides whether a failed primary-backend boot should *latch* into
 * `backendStartFailure`. A latched failure makes every subsequent
 * startHermes() re-throw the cached error without re-attempting the connect —
 * the right behavior for a LOCAL backend so the renderer's retry loop can't
 * restart a broken install over and over.
 *
 * It is the WRONG behavior for a REMOTE backend. A remote connect can fail for
 * transient reasons — a lapsed OAuth access-token cookie (the gateway rotates a
 * fresh one from the live refresh-token cookie on the next request), a
 * ws-ticket mint that timed out mid sleep/wake, or a host that was briefly
 * unreachable across a laptop sleep. There is no child process whose 'exit'
 * handler would clear the cache, so a latched remote failure sticks until the
 * whole app is quit and relaunched: reconnect, "Sign out & sign in" (which only
 * reloads the renderer), and the wake-recovery revalidate path all keep hitting
 * the same stale error. Not latching lets the very next connect re-mint a
 * ticket against the (now refreshed) session and self-heal.
 *
 * Extracted as a dependency-free pure predicate so the invariant is testable
 * without booting Electron or reading main.ts source text.
 */

export interface BackendStartFailureContext {
  /**
   * True when the boot that just failed was resolving/dialing a REMOTE (or
   * cloud) primary backend rather than spawning a local child.
   */
  attemptedRemote: boolean
}

/**
 * Whether a startHermes() failure should latch into `backendStartFailure`.
 * Latch local failures (prevent install-restart loops); never latch remote
 * failures (they are transient and must stay retryable so recovery paths work
 * without an app restart).
 */
export function shouldLatchBackendStartFailure(context: BackendStartFailureContext): boolean {
  return !context.attemptedRemote
}

export interface RemoteReauthFailureContext {
  /** True when the boot that just failed was dialing a REMOTE (or cloud) backend. */
  attemptedRemote: boolean
  /**
   * True when the failure was a CONFIRMED auth rejection (a credentialed
   * probe got 401/403), not a transient connectivity fault.
   */
  isReauth: boolean
}

/**
 * Whether a failed remote boot should latch as a reauth failure.
 *
 * This is the deliberate counterpart to `shouldLatchBackendStartFailure`,
 * which never latches a remote failure because remote faults are usually
 * transient and must stay retryable. A *confirmed* reauth rejection is the
 * exception: it cannot self-heal, because nothing will change until the user
 * signs in again.
 *
 * Without a latch, the non-latching remote path actively prevents recovery.
 * Every subsequent `getConnection`/`api` call re-runs `startHermes`, re-emits
 * `running: true`, and the boot-failure overlay (`visible = Boolean(boot.error)
 * && !boot.running`) hides itself — so the "Sign in" button flickers out from
 * under the user before they can click it. Latching holds the overlay still
 * and clickable. Cleared on every recovery path (reset, repair, apply-config,
 * and a confirmed sign-in) so a fresh session boots normally.
 */
export function shouldLatchRemoteReauthFailure(context: RemoteReauthFailureContext): boolean {
  return context.attemptedRemote && context.isReauth
}

/**
 * The one LOCAL failure class that cannot self-heal across restarts: the
 * backend is HTTP-reachable (so the install is healthy) but /api/ws rejects
 * the session token. A stale HERMES_DASHBOARD_SESSION_TOKEN in the Hermes
 * .env overrides the token the desktop injects into its own spawned backend,
 * so every restart fails identically and the renderer's "Reload and retry"
 * loop spins forever. Matching the exact message both WS-probe throw sites in
 * main.ts construct keeps unrelated failures (port timeout, child exit,
 * connection-test WS probes) out of this class.
 */
export const MAX_TOKEN_REJECTION_BOOT_RETRIES = 3

export function isSessionTokenRejectionError(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error)

  return message.includes('WebSocket (/api/ws) rejected the session token')
}

/**
 * Actionable hint appended to the surfaced failure so the user knows the fix
 * instead of watching a silent retry loop. `envPath` is the resolved Hermes
 * .env (the main process passes HERMES_HOME/.env).
 */
export function sessionTokenRejectionHint(envPath: string): string {
  return (
    `A stale HERMES_DASHBOARD_SESSION_TOKEN= line in ${envPath} can override the token the desktop ` +
    'injects into its own backend and cause exactly this rejection. Remove that line (or run ' +
    '`hermes setup`) and retry.'
  )
}

export interface TokenRejectionRetryGuard {
  /** Consecutive WS session-token rejections seen so far. */
  readonly count: number
  /** True once `count` has reached the bound; the reset loop must stop. */
  readonly exhausted: boolean
  /**
   * Record one boot failure. Token rejections extend the streak; any other
   * failure class breaks it (the bound only applies to CONSECUTIVE
   * rejections, so unrelated failures keep their existing retry behavior).
   */
  recordFailure(error: unknown): void
  /** Called on a clean boot completion so the next episode starts fresh. */
  reset(): void
}

export function createTokenRejectionRetryGuard(
  maxRetries: number = MAX_TOKEN_REJECTION_BOOT_RETRIES
): TokenRejectionRetryGuard {
  let consecutive = 0

  return {
    get count() {
      return consecutive
    },

    get exhausted() {
      return consecutive >= maxRetries
    },

    recordFailure(error) {
      consecutive = isSessionTokenRejectionError(error) ? consecutive + 1 : 0
    },

    reset() {
      consecutive = 0
    }
  }
}
