/**
 * Liveness-probe force-close policy for the primary gateway socket (#95327).
 *
 * Why this exists
 * ───────────────
 * Every window-lifecycle recovery signal (power resume, network online,
 * focus, visibility) nudges `reconnectNow()`. When the socket still reports
 * open, that path probes liveness with a short-bounded ping and FORCE-CLOSES
 * the socket when the ping times out — "a half-open TCP connection must not
 * swallow the user's next submit".
 *
 * A ping timeout proves a dead TRANSPORT, but it also fires for a live,
 * BUSY backend: a long silent tool call (a quiet build, an OCR worker, a
 * large download) starves the gateway's event loop past the 5s probe budget
 * (#74874's GIL-stall family, amplified on Windows by AV/filter-driver
 * latency). Tearing down the renderer↔backend WebSocket at that moment is a
 * false kill: the gateway sees its client vanish mid-turn, the
 * `ws_orphan_reap` grace expires, and the running turn is interrupted into a
 * bare "Operation interrupted." placeholder — the exact #95327 report.
 *
 * Policy
 * ──────
 * While ANY session still reports working (`$workingSessionIds`), backend
 * silence is EXPECTED, so a probe timeout is inconclusive rather than proof
 * of death. The first such timeout is deferred: keep the socket and schedule
 * one bounded re-probe instead. Only when the failure STREAK reaches
 * `LIVENESS_PROBE_FAILURE_STREAK` (the re-probe also went unanswered — or no
 * turn is in flight at all) do we treat the socket as genuinely dead and let
 * the normal close→backoff→redial machinery take over. Worst case this adds
 * one re-probe interval to genuine-dead detection; it removes the mid-turn
 * teardown entirely for busy-but-healthy backends.
 *
 * Pure and Electron-free so the streak boundaries are assertable directly
 * (mirroring gateway-liveness usage in use-gateway-boot).
 */

/** Consecutive unanswered probes tolerated while work is in flight. */
export const LIVENESS_PROBE_FAILURE_STREAK = 2

/** How long after a deferred probe we try again (bounded, coalesced). */
export const LIVENESS_REPROBE_DELAY_MS = 3_000

export interface LivenessForceCloseInput {
  /**
   * How many sessions currently report working (mid-turn). Zero means no
   * turn is riding this socket, so silence has no innocent explanation.
   */
  workingSessionCount: number
  /**
   * Length of the CURRENT unanswered-probe streak, INCLUDING the failure
   * being decided (first failure = 1).
   */
  consecutiveFailures: number
}

export type LivenessForceCloseReason = 'in-flight-work-deferred' | 'failure-streak-exhausted' | 'no-in-flight-work'

export interface LivenessForceCloseDecision {
  close: boolean
  reason: LivenessForceCloseReason
}

/**
 * Decide whether one liveness-probe failure should force the socket down.
 *
 * - No work in flight            → close immediately (unchanged legacy shape:
 *                                  a dead idle socket buys nothing by waiting).
 * - Work in flight, streak < max → keep the socket; the caller schedules a
 *                                  bounded re-probe ('in-flight-work-deferred').
 * - Work in flight, streak ≥ max → close anyway ('failure-streak-exhausted'):
 *                                  a persistently unresponsive socket must
 *                                  still be rebuilt, never trusted forever.
 */
export function decideLivenessForceClose(input: LivenessForceCloseInput): LivenessForceCloseDecision {
  const workingSessionCount = Math.max(0, Math.floor(input.workingSessionCount))
  const consecutiveFailures = Math.max(1, Math.floor(input.consecutiveFailures))

  if (workingSessionCount > 0 && consecutiveFailures < LIVENESS_PROBE_FAILURE_STREAK) {
    return { close: false, reason: 'in-flight-work-deferred' }
  }

  return workingSessionCount > 0
    ? { close: true, reason: 'failure-streak-exhausted' }
    : { close: true, reason: 'no-in-flight-work' }
}
