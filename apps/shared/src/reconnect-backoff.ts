/**
 * Exponential reconnect backoff shared by every Hermes front-end socket
 * (desktop gateway/plugins, web events feed, web PTY, Ink TUI attach mode).
 *
 * Default is full jitter: a bare exponential backoff still lets every client
 * in a fleet retry in lockstep — after a gateway restart, N clients that all
 * disconnected within the same instant all wake up and redial at the same
 * instant too, which is a reconnect storm by another name. Full jitter (AWS's
 * "Exponential Backoff And Jitter") spreads that out: each attempt sleeps a
 * *random* duration between 0 and the exponential ceiling.
 *
 * https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/
 *
 * `jitter: false` returns the ceiling itself — the deterministic ladder the
 * web dashboard renders into its "reconnecting in Ns" banner.
 */

export interface ReconnectBackoffOptions {
  /** Delay for the first retry (attempt 0) before jitter is applied, in ms. */
  baseDelayMs?: number
  /** Ceiling on the exponential delay before jitter is applied, in ms. */
  capMs?: number
  /** Full jitter (default) or the bare ceiling. */
  jitter?: boolean
}

const DEFAULT_BASE_DELAY_MS = 300
const DEFAULT_CAP_MS = 15_000
// 2 ** attempt overflows to Infinity long before it matters (attempt would
// need to be ~1024) and Math.min against a finite cap keeps the ceiling sane
// regardless; the clamp just keeps the arithmetic in finite territory.
const MAX_EXPONENT = 32

/**
 * Delay before reconnect attempt number `attempt` (0-indexed: the first retry
 * after the initial failure is `attempt = 0`). With jitter the value lies in
 * `[0, min(capMs, baseDelayMs * 2 ** attempt))`; without it, it IS that ceiling.
 */
export function reconnectBackoffDelayMs(attempt: number, options: ReconnectBackoffOptions = {}): number {
  const baseDelayMs = options.baseDelayMs ?? DEFAULT_BASE_DELAY_MS
  const capMs = options.capMs ?? DEFAULT_CAP_MS
  const exponent = Math.min(Math.max(0, Math.trunc(attempt)), MAX_EXPONENT)
  const ceiling = Math.min(capMs, baseDelayMs * 2 ** exponent)

  return options.jitter === false ? ceiling : Math.random() * ceiling
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
