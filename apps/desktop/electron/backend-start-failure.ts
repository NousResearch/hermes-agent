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
