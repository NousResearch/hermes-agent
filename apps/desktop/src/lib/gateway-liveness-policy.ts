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
 * Policy — one principle, "real service proof", in both directions
 * ────────────────────────────────────────────────────────────────
 * Silence is inconclusive while the backend has a legitimate reason to be
 * silent, and a socket is only provably ALIVE when it delivers frames:
 *
 * 1. In-flight work makes silence inconclusive. Both a running turn
 *    (`$workingSessionIds`) and a pending RPC (a `session.resume` grinding
 *    through a cold agent build, a slow sidebar refresh) legitimately starve
 *    the event loop past the probe budget. Tearing the socket down then
 *    cancels the very work being waited on and redials into the same busy
 *    backend — each redial firing fresh full-view refreshes, piling on more
 *    load: the observed reconnect storms.
 * 2. Delivered frames are proof of life. If the socket served ANY frame
 *    within `LIVENESS_SERVED_GRACE_MS` (an RPC response, an event delta, a
 *    heartbeat reply), the transport is demonstrably alive and a starved
 *    probe must not kill it, however long the unanswered-probe streak.
 * 3. Without either, the streak rules. The first timeout with work in
 *    flight defers behind one bounded re-probe; a streak of
 *    `LIVENESS_PROBE_FAILURE_STREAK` unanswered probes with no frame
 *    delivered — or no in-flight work at all — closes the socket, so a
 *    genuinely dead transport (half-open after sleep) is still rebuilt
 *    within seconds, never left to a pending RPC's own multi-minute timeout.
 *
 * Pure and Electron-free so the boundaries are assertable directly
 * (mirroring gateway-liveness usage in use-gateway-boot).
 */

/** Consecutive unanswered probes tolerated while work is in flight. */
export const LIVENESS_PROBE_FAILURE_STREAK = 2

/** How long after a deferred probe we try again (bounded, coalesced). */
export const LIVENESS_REPROBE_DELAY_MS = 3_000

/**
 * A frame delivered within this window is proof of life: the transport
 * serves, the probe merely starved. Far above the worst measured single
 * event-loop stall bursts (10-15s REST fan-outs, response-serialization
 * grinds), far below the multi-minute RPC timeouts a dead socket must never
 * be left to. Only consulted while work is in flight — an idle dead socket
 * still closes immediately.
 */
export const LIVENESS_SERVED_GRACE_MS = 60_000

export interface LivenessForceCloseInput {
  /**
   * How many sessions currently report working (mid-turn). Zero means no
   * turn is riding this socket.
   */
  workingSessionCount: number
  /**
   * Length of the CURRENT unanswered-probe streak, INCLUDING the failure
   * being decided (first failure = 1).
   */
  consecutiveFailures: number
  /**
   * In-flight RPCs still awaiting a response on this socket (a pending
   * session.resume, a sidebar refresh). Like a running turn, a pending RPC
   * makes probe silence inconclusive — the caller is already waiting on the
   * backend, bounded by that RPC's own timeout. Omitted = 0.
   */
  pendingRpcCount?: number
  /**
   * Milliseconds since the socket last delivered a parsed frame; Infinity
   * (or omitted) when this generation has served nothing / the caller has no
   * proof. Recent delivery is positive proof of life (rule 2 above).
   */
  msSinceLastServed?: number
}

export type LivenessForceCloseReason =
  | 'in-flight-work-deferred'
  | 'recent-service-deferred'
  | 'failure-streak-exhausted'
  | 'no-in-flight-work'

export interface LivenessForceCloseDecision {
  close: boolean
  reason: LivenessForceCloseReason
}

/**
 * Decide whether one liveness-probe failure should force the socket down.
 *
 * - No work in flight (no turn, no pending RPC)
 *                                → close immediately (unchanged legacy shape:
 *                                  a dead idle socket buys nothing by waiting).
 * - Work in flight, frame served within the grace window
 *                                → keep the socket ('recent-service-deferred'):
 *                                  delivered frames are proof of life; the
 *                                  probe starved, the transport did not die.
 * - Work in flight, streak < max → keep the socket; the caller schedules a
 *                                  bounded re-probe ('in-flight-work-deferred').
 * - Work in flight, streak ≥ max, nothing served
 *                                → close anyway ('failure-streak-exhausted'):
 *                                  a persistently silent socket must still be
 *                                  rebuilt, never trusted forever.
 */
export function decideLivenessForceClose(input: LivenessForceCloseInput): LivenessForceCloseDecision {
  const sanitizeCount = (value: number | undefined) =>
    Number.isFinite(value) ? Math.max(0, Math.floor(value as number)) : 0

  const workingSessionCount = sanitizeCount(input.workingSessionCount)
  const consecutiveFailures = Math.max(1, sanitizeCount(input.consecutiveFailures))
  const pendingRpcCount = sanitizeCount(input.pendingRpcCount)

  // NaN must read as "no proof" (Infinity), never as "served just now".
  const msSinceLastServed = Number.isFinite(input.msSinceLastServed)
    ? (input.msSinceLastServed as number)
    : Number.POSITIVE_INFINITY

  if (workingSessionCount === 0 && pendingRpcCount === 0) {
    return { close: true, reason: 'no-in-flight-work' }
  }

  if (msSinceLastServed < LIVENESS_SERVED_GRACE_MS) {
    return { close: false, reason: 'recent-service-deferred' }
  }

  if (consecutiveFailures < LIVENESS_PROBE_FAILURE_STREAK) {
    return { close: false, reason: 'in-flight-work-deferred' }
  }

  return { close: true, reason: 'failure-streak-exhausted' }
}
