// Pure decision helpers for the zombie-stream (half-open gRPC) watchdog.
//
// spectrum-ts only reconnects when its inbound async iterator throws or ends.
// A half-open ("zombie") socket makes the iterator hang forever — no error,
// no end — so inbound silently dies while /healthz still looks fine. The
// watchdog in index.mjs tracks the last time the inbound iterator yielded and,
// once the stream has been silent past a conservative threshold, drives a
// cheap authenticated unary read over the same channel. STRICT semantics:
//
//   - probe resolves, or rejects with a SERVER-ANSWERED error for our
//     synthetic id (not-found, INVALID_ARGUMENT, any `[spectrum-*]`-stamped
//     SDK error)              -> ALIVE (the wire round-tripped)
//   - probe rejects any other way (UNAVAILABLE, DEADLINE_EXCEEDED, network
//     down, FAILED_PRECONDITION, ...) -> INCONCLUSIVE — never treated as
//                              alive, and never treated as zombie-proof either
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

// gRPC codes/text that prove a SERVER answered our probe: the round-trip
// completed, so the channel is alive even though the probe (a synthetic
// message id) was rejected. Anything else is inconclusive.
//   5 / "notFound"       — NOT_FOUND: synthetic id doesn't exist
//   3 / "invalidArgument"— INVALID_ARGUMENT: Spectrum parses the probe id and
//                          rejects it (e.g. "Expected message resource GUID");
//                          the server answered, so the wire round-tripped.
// FAILED_PRECONDITION is deliberately NOT here: it signals a broken local
// state (line unprovisioned, key rotated), not a healthy channel.
const NOT_FOUND_RE = /not[\s_-]?found/i;
const SERVER_ANSWERED_RE = /invalid[\s_-]?argument/i;
const SERVER_STAMPED_RE = /\[spectrum-[\w-]+\]/;
// Fail-closed guard: transport/connection failures and FAILED_PRECONDITION
// prove nothing about the wire round-trip, so they stay inconclusive even when
// the SDK also stamps them (`[spectrum-imessage] ...`) or the text mentions
// "invalid argument". Checked BEFORE the server-answered rules.
const NO_ROUND_TRIP_RE =
  /failed[\s_-]?precondition|unavailable|deadline|econnrefused|econnreset|etimedout|enetunreach|epipe|socket hang up|fetch failed|timed?[\s-]?out|tls|certificate/i;
/** Return a unique message id accepted by Spectrum's message-id parser. */
export function createProbeMessageId() {
  return randomUUID();
}

/**
 * Classify the rejection of the synthetic-id probe read.
 *
 * A rejection still proves the wire round-tripped when the SERVER produced it:
 * not-found, INVALID_ARGUMENT (the gRPC numeric code, the SDK's
 * "invalidArgument" string, or that message text), or any error carrying an
 * SDK `[spectrum-*]` provenance stamp. Transport failures and
 * FAILED_PRECONDITION stay inconclusive.
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
  const noRoundTrip =
    code === 14 ||
    code === 4 ||
    code === 9 ||
    code === "unavailable" ||
    code === "deadlineExceeded" ||
    code === "failedPrecondition" ||
    NO_ROUND_TRIP_RE.test(message);
  if (noRoundTrip) {
    // Transport dead / precondition broken — nothing was proven either way.
    return { alive: false, inconclusive: true, reason: message };
  }
  const serverAnswered =
    code === 5 ||
    code === 3 ||
    code === "notFound" ||
    code === "invalidArgument" ||
    NOT_FOUND_RE.test(message) ||
    SERVER_ANSWERED_RE.test(message) ||
    SERVER_STAMPED_RE.test(message);
  if (serverAnswered) {
    // Expected: the synthetic id can't be resolved, but the server answered.
    // The unary call completed a round-trip, so the channel is provably alive.
    return { alive: true, inconclusive: false, reason: "server-answered round-trip" };
  }
  // Anything else (UNAVAILABLE, DEADLINE_EXCEEDED, FAILED_PRECONDITION, TLS,
  // network down, ...) does NOT prove liveness — and doesn't prove a zombie
  // either.
  return { alive: false, inconclusive: true, reason: message };
}

/**
 * Explicit readiness for the inbound stream: true ONLY for a stream that is
 * currently `healthy`. ``starting`` (never connected yet), ``recovering``
 * (reconnect backoff in progress) and ``degraded`` are all NOT ready. Mirrors
 * the existing ``/healthz`` `ok` flag (``state !== "degraded"``) without
 * changing it — out-of-process senders and the Python adapter keep consuming
 * the existing fields.
 *
 * @param {string} state one of "starting" | "healthy" | "recovering" | "degraded"
 * @returns {boolean}
 */
export function isStreamReady(state) {
  return state === "healthy";
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
