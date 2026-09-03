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

export interface RemoteBootRetryContext {
  /** True when the boot that just failed was dialing a REMOTE (or cloud/SSH) backend. */
  attemptedRemote: boolean
  /**
   * True when the failure was a CONFIRMED auth rejection (401/403), which can
   * never self-heal without the user signing in again.
   */
  isReauth: boolean
  /**
   * True when SSH refused to connect because the host's key CHANGED
   * (StrictHostKeyChecking fails closed). Retrying cannot succeed until the
   * user verifies the change and removes the stale known_hosts entry, so this
   * is terminal like a reauth rejection — not connectivity.
   */
  isHostKeyChanged?: boolean
}

/**
 * A host-key-change refusal is identifiable both by the `kind` tag
 * classifySshError puts on the error and — for errors that crossed a
 * stringifying boundary — by the stable phrases ssh/our own message carry.
 * One user hit 157 consecutive boot-retry failures over 2.5h against a
 * reinstalled VPS (Aug 2026 bundle) because this was classified as transient.
 */
export function isHostKeyChangedBootFailure(error: unknown): boolean {
  if ((error as { kind?: string } | null | undefined)?.kind === 'host-key-changed') {
    return true
  }

  const message = error instanceof Error ? error.message : String(error ?? '')

  return /REMOTE HOST IDENTIFICATION HAS CHANGED|Host key verification failed|host key for .+ has CHANGED/i.test(
    message
  )
}

/**
 * Whether a failed remote boot should latch (into `backendStartFailure`)
 * because the host key changed. Same rationale as the reauth latch: the
 * failure cannot self-heal, and an unlatched terminal failure makes every
 * recovery surface re-drive the identical doomed boot. The latch is released
 * by the existing reset/repair/apply-config paths once the user has run
 * `ssh-keygen -R <host>`.
 */
export function shouldLatchHostKeyChangedFailure(context: RemoteBootRetryContext): boolean {
  return context.attemptedRemote && context.isHostKeyChanged === true
}

/**
 * Whether a failed primary-backend boot is a TRANSIENT remote failure the
 * renderer may retry automatically (bounded, with backoff).
 *
 * This closes the self-heal gap of issue #82679: a dropped SSH/HTTP remote
 * connection surfaces at the next boot as a transient transport failure
 * ("Could not verify the existing SSH backend", ERR_CONNECTION_RESET, mint
 * timeouts). Those never latch (see shouldLatchBackendStartFailure), but
 * nothing ever RE-ATTEMPTED the boot either — the renderer's reconnect loop
 * only arms after a completed boot, so the app sat on "Desktop boot failed"
 * until the user manually re-entered the same connection details (which just
 * forced a fresh bootstrap). A missing capability differs from a transient
 * failure: confirmed reauth rejections, host-key changes, and local failures
 * stay out of the retry path; everything else remote is connectivity and
 * should retry.
 */
export function isRetryableRemoteBootFailure(context: RemoteBootRetryContext): boolean {
  return context.attemptedRemote && !context.isReauth && context.isHostKeyChanged !== true
}
