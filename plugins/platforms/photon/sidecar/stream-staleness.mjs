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
//     synthetic id            -> ALIVE (the wire round-tripped)
//   - probe rejects with a server-answered error — gRPC INVALID_ARGUMENT,
//     or a message stamped with a spectrum component source like
//     "[spectrum-imessage] ..." -> ALIVE (a server-side answer also proves
//     the wire round-tripped; the synthetic probe id is not a valid Apple
//     message GUID, so the proxy rejects it with a validation error on
//     every probe — #101618)
//   - probe rejects any other way (UNAVAILABLE, DEADLINE_EXCEEDED, network
//     down, ...)              -> INCONCLUSIVE — never treated as alive, and
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

// gRPC NOT_FOUND is code 5; SDKs also surface it as "not found" / "NotFound"
// message text. Anything not clearly not-found is inconclusive.
const NOT_FOUND_RE = /not[\s_-]?found/i;

// A server-answered rejection also proves a round-trip: gRPC
// INVALID_ARGUMENT (3) is generated server-side, as are errors stamped
// with a spectrum component source — the "[spectrum-*]" prefixes do not
// exist anywhere in the installed client tree (#101618), so a message
// carrying one provably came back over the wire. FAILED_PRECONDITION (9)
// is deliberately excluded: the proxy can also answer it from a
// half-ready state that never completed the probe.
const SERVER_ANSWERED_CODES = new Set([3, "invalidArgument"]);
const SERVER_ANSWERED_RE = /\[spectrum-[a-z0-9-]+\]/i;

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
  if (SERVER_ANSWERED_CODES.has(code) || SERVER_ANSWERED_RE.test(message)) {
    // The server received the request and answered with an application-level
    // rejection (the synthetic id is not a valid Apple message GUID, so the
    // proxy rejects it as a validation error instead of not-found). A
    // server-side answer proves the wire round-tripped just the same.
    return {
      alive: true,
      inconclusive: false,
      reason: "server-answered round-trip: " + message,
    };
  }
  // Anything else (UNAVAILABLE, DEADLINE_EXCEEDED, TLS, auth, ...) does NOT
  // prove liveness — and doesn't prove a zombie either.
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
