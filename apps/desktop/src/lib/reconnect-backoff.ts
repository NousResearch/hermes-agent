/**
 * Full-jitter exponential backoff for gateway WebSocket reconnects.
 *
 * A bare exponential backoff still lets every renderer in a fleet retry in
 * lockstep — after a gateway restart (e.g. following an update), N desktop
 * clients that all disconnected within the same instant all wake up and
 * redial at the same instant too, which is a reconnect storm by another
 * name. Full jitter (AWS's "Exponential Backoff And Jitter") spreads that
 * out: each attempt sleeps a *random* duration between 0 and the exponential
 * ceiling, so retries desynchronize instead of pulsing together.
 *
 * https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/
 */

export interface ReconnectBackoffOptions {
  /** Ceiling on the exponential delay before jitter is applied, in ms. */
  capMs?: number
  /** Delay for the first retry (attempt 0) before jitter is applied, in ms. */
  baseDelayMs?: number
}

const DEFAULT_BASE_DELAY_MS = 300
const DEFAULT_CAP_MS = 15_000

/**
 * Delay before reconnect attempt number `attempt` (0-indexed: the first
 * retry after the initial failure is `attempt = 0`). Returns a value in
 * `[0, min(capMs, baseDelayMs * 2 ** attempt))` — full jitter, not
 * "equal jitter" or "decorrelated jitter", so it can occasionally return a
 * very small delay even at a high attempt count. That's intentional: it's
 * the variant with the best-documented storm-avoidance behavior and no
 * accumulated-delay state to track between calls.
 */
export function reconnectBackoffDelayMs(attempt: number, options: ReconnectBackoffOptions = {}): number {
  const baseDelayMs = options.baseDelayMs ?? DEFAULT_BASE_DELAY_MS
  const capMs = options.capMs ?? DEFAULT_CAP_MS
  const safeAttempt = Math.max(0, attempt)

  // 2 ** attempt overflows to Infinity long before it matters (attempt would
  // need to be ~1024), and Math.min against a finite cap keeps the ceiling
  // sane regardless, so no extra clamping is needed here.
  const ceiling = Math.min(capMs, baseDelayMs * 2 ** safeAttempt)

  return Math.random() * ceiling
}

/**
 * Attempt counter to carry into the next reconnect schedule after a socket
 * closed.
 *
 * The counter resets only on REAL SERVICE PROOF: the closing socket
 * generation delivered at least one parsed JSON-RPC frame (the gateway.ready
 * readiness ack, an RPC response, an event). Resetting on `state === 'open'`
 * alone defeats the backoff entirely against a backend that ACCEPTS sockets
 * but cannot serve them — a cold boot grinding through its skill scan, or a
 * starved event loop. Every dial "succeeds", resets the counter, dies
 * seconds later, and redials at the attempt-0 ceiling: a sub-second
 * reconnect storm that both hammers the struggling backend and re-cancels
 * whatever it was loading (#94769's flicker family; the
 * "conn:local::default dial failed" storms).
 *
 * A temporal heuristic (socket stayed open N seconds) is NOT proof either:
 * a degraded-but-open socket outlives any window without ever serving, and
 * would reset the ladder right back into the amplification it exists to
 * stop. Service proof is binary and observable — the transport either
 * delivered a frame this generation or it did not.
 */
export function reconnectAttemptAfterClose(previousAttempt: number, servedProof: boolean): number {
  const safePrevious = Number.isFinite(previousAttempt) ? Math.max(0, Math.floor(previousAttempt)) : 0

  return servedProof ? 0 : safePrevious
}
