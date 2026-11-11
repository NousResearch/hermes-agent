// Pure decision helpers for the zombie-stream (half-open gRPC) watchdog.
//
// spectrum-ts only reconnects when its inbound async iterator throws or ends.
// A half-open ("zombie") socket makes the iterator hang forever — no error,
// no end — so inbound silently dies while /healthz still looks fine. The
// watchdog in index.mjs tracks the last time the inbound iterator yielded and,
// once the stream has been silent past a conservative threshold, drives a
// cheap authenticated unary read over the same channel. STRICT semantics:
//
//   - probe resolves, or rejects with a not-found-shaped error for our
//     synthetic id, or with a policy rejection answered by the server
//                             -> ALIVE (the wire round-tripped)
//   - probe rejects any other way (UNAVAILABLE, DEADLINE_EXCEEDED, network
//     down, local SDK validation, ...)
//                             -> INCONCLUSIVE — never treated as alive, and
//                                never treated as zombie-proof either
//
// A zombie is only declared when the stream is silent past the threshold AND
// a probe proves connectivity (the wire works but the stream is deaf). Silence
// alone NEVER degrades the stream: shared lines can be legitimately quiet for
// hours. Inconclusive probes NEVER degrade it either: the network may simply
// be down, and in that case the iterator will eventually throw and the
// existing re-subscribe loop recovers on its own.
//
// These helpers are pure (no SDK, no timers) so tests can execute them under
// node — see tests/plugins/platforms/photon/test_zombie_stream_watchdog.py.

import { randomUUID } from "node:crypto";

// gRPC NOT_FOUND is code 5; SDKs also surface it as "not found" / "NotFound"
// message text. Anything not clearly not-found is inconclusive.
const NOT_FOUND_RE = /not[\s_-]?found/i;
// Shared-pool (Free/Pro) plans only allow targets registered as project
// users, so the default synthetic probe target is answered by the server with
// a policy rejection ("Target not allowed for this project") instead of
// not-found. That answer still proves the round-trip completed.
const POLICY_REJECTION_RE = /target[\s_-]?not[\s_-]?allowed/i;
// gRPC PERMISSION_DENIED (7) / UNAUTHENTICATED (16) can only be answered by
// the server; the local SDK never raises them before the request hits the
// wire (INVALID_ARGUMENT stays excluded: the SDK validator raises it locally
// for a malformed probe id, and that must not fake liveness).
const SERVER_ANSWERED_CODES = new Set([7, 16, "permissionDenied", "unauthenticated"]);
/** Return a unique message id accepted by Spectrum's message-id parser. */
export function createProbeMessageId() {
  return `spc-msg-${randomUUID()}`;
}

/**
 * Classify the rejection of the synthetic-id probe read.
 *
 * @param {unknown} err error thrown by `space.getMessage(<synthetic id>)`
 * @returns {{alive: boolean, inconclusive: boolean, reason: string}}
 */
export function classifyProbeRejection(err) {
  const code = err && typeof err === "object" ? err.code : undefined;
  const message =
    err && typeof err === "object" && err.message
      ? String(err.message)
      : String(err);
  if (code === 5 || code === "notFound" || NOT_FOUND_RE.test(message)) {
    // Expected: the synthetic id doesn't exist. The unary call completed a
    // round-trip, so the channel is provably alive.
    return { alive: true, inconclusive: false, reason: "not-found round-trip" };
  }
  if (SERVER_ANSWERED_CODES.has(code) || POLICY_REJECTION_RE.test(message)) {
    // The server itself rejected the probe (plan allowlist, permissions) —
    // the wire round-tripped, so the channel is alive all the same.
    return {
      alive: true,
      inconclusive: false,
      reason: "server-answered policy rejection",
    };
  }
  // Anything else (UNAVAILABLE, DEADLINE_EXCEEDED, TLS, local SDK
  // validation, ...) does NOT prove liveness — and doesn't prove a zombie
  // either.
  return { alive: false, inconclusive: true, reason: message };
}

/**
 * Should the watchdog probe at all this tick?
 *
 * @param {number} silentForMs   ms since the inbound iterator last yielded
 * @param {number} thresholdMs   silence threshold (<= 0 disables the watchdog)
 * @param {number} sinceLastProbeMs ms since the previous probe attempt
 * @param {number} probeCooldownMs  min spacing between probe attempts
 * @returns {boolean}
 */
export function shouldProbe(silentForMs, thresholdMs, sinceLastProbeMs, probeCooldownMs) {
  if (!(thresholdMs > 0)) return false;
  if (silentForMs < thresholdMs) return false;
  return sinceLastProbeMs >= probeCooldownMs;
}

/**
 * Final classification: zombie only on silence past threshold + probe-proven
 * connectivity. Never on silence alone, never on an inconclusive probe.
 *
 * @param {number} silentForMs ms since the inbound iterator last yielded
 * @param {number} thresholdMs silence threshold (<= 0 disables the watchdog)
 * @param {{alive: boolean}} probeOutcome
 * @returns {boolean}
 */
export function isZombieSuspect(silentForMs, thresholdMs, probeOutcome) {
  if (!(thresholdMs > 0)) return false;
  if (silentForMs < thresholdMs) return false;
  return probeOutcome != null && probeOutcome.alive === true;
}
