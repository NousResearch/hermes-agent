/**
 * WHEN the window's primary socket arms its next retry (#C5).
 *
 * The delay, the timer and the attempt count belong to the window's connection
 * supervisor (store/gateway-supervisor), which keeps at most ONE ladder per
 * route for the whole window. Nothing here keeps state: this is only the
 * decision — the set of gates that must all say "yes" before a rung is armed —
 * extracted from the boot effect, which sits at its own line cap (the repo's
 * code-health ratchet, FUNC_LINES).
 *
 * Kept as ports rather than as reads of the effect's locals so the policy is
 * testable on its own and the effect cannot silently grow another retry path.
 */
import { armRouteRetry, cancelRouteRetry, PRIMARY_ROUTE, routeRetryPending } from '@/store/gateway-supervisor'

/** Where the next attempt continues from, when the caller has one. */
export interface PrimaryRetryResume {
  activationEpoch: number
  profile: string
}

export interface PrimaryRetryGate {
  /** The effect was torn down: never arm again. */
  cancelled: () => boolean
  /** A dial is already running. */
  dialing: () => boolean
  /** The socket is open — there is nothing to reconnect to. */
  open: () => boolean
  /** The window is parked: a hidden window dials nothing (#C2a). */
  parked: () => boolean
  /** A confirmed reauth rejection; only the user's sign-in clears it. */
  reauthBlocked: () => boolean
  /** A deliberate gateway switch owns this window's socket right now. */
  switching: () => boolean
}

export interface PrimaryRetryLadder {
  /** Arm the window's next attempt, or refuse while a rung is already armed. */
  arm: (resume?: PrimaryRetryResume) => void
  /** Disarm the armed attempt: a wake/recovery signal dials instead. */
  cancel: () => void
}

export function createPrimaryRetryLadder(
  gate: PrimaryRetryGate,
  attempt: (resume?: PrimaryRetryResume) => void
): PrimaryRetryLadder {
  return {
    arm: (resume?: PrimaryRetryResume) => {
      if (
        gate.cancelled() ||
        gate.parked() ||
        gate.reauthBlocked() ||
        gate.dialing() ||
        routeRetryPending(PRIMARY_ROUTE) ||
        gate.open() ||
        gate.switching()
      ) {
        return
      }

      armRouteRetry(PRIMARY_ROUTE, () => attempt(resume))
    },
    cancel: () => cancelRouteRetry(PRIMARY_ROUTE)
  }
}
