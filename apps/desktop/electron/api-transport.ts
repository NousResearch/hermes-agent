/**
 * Shared HTTP transport policy for the Electron main process's Hermes REST
 * helpers (fetchJson / fetchPublicJson / downloadViaTokenToFile).
 *
 * Two concerns live here so they can be unit-tested without Electron:
 *
 * 1. Connection-pooled keep-alive agents. Opening a fresh TCP socket per call
 *    is what produced the burst-traffic ECONNRESET storms (#92976): the
 *    backend closes idle keep-alive sockets and the next write on a reused
 *    raw socket dies with 'socket hang up'. JSON calls and streaming
 *    downloads get SEPARATE pools so a handful of long-lived download
 *    streams can never starve the small, latency-sensitive JSON calls out of
 *    the socket pool.
 *
 * 2. A retry policy that is safe for non-idempotent verbs. A transient
 *    transport error does NOT mean the server didn't process the request —
 *    an ECONNRESET can arrive after the backend already handled a POST
 *    (created the session, submitted the prompt) and merely lost the socket
 *    before the response was read. Blindly retrying every verb double-submits.
 *
 *    The rule implemented by shouldRetryRequest():
 *      - Idempotent verbs (GET / HEAD / OPTIONS) retry on any transient
 *        transport error — replaying them is harmless by definition.
 *      - Non-idempotent verbs (POST / PUT / PATCH / DELETE) retry ONLY when
 *        the request provably never reached the server:
 *          a) connection-establishment failures (ECONNREFUSED, ENOTFOUND,
 *             EAI_AGAIN, EHOSTUNREACH, ENETUNREACH) — no connection means no
 *             request; or
 *          b) a transient error thrown before we started flushing the
 *             request (requestState.bodySent === false).
 *        Anything ambiguous — ECONNRESET / EPIPE / 'socket hang up' after
 *        the body went out — is NOT retried; the error surfaces to the
 *        caller. When in doubt, don't retry a non-idempotent request.
 */

import http from 'node:http'
import https from 'node:https'

// JSON pool: many small concurrent calls (session lists, config, prompts).
//
// maxSockets is deliberately NOT the typical direct-TCP value (50+). When the
// transport is an SSH mux (remote backends), all logical sockets multiplex
// over ONE physical SSH connection: the mux serialises channel handling, so
// ~10 concurrent channels is the observed saturation point. Exceeding it
// queues requests behind each other, turning a healthy 0.3s round-trip into
// 4-9s latency that the retry policy then amplifies into a connection storm
// (conns 10 to 22, channels 3 to 11, self-resolving after ~50s). 12 leaves
// headroom above the healthy steady-state (~10 conns) while staying below
// the mux saturation knee, and is equally safe for direct TCP (HTTP JSON
// payloads are small; 12 concurrent sockets suffice for interactive rates).
const MAX_JSON_CONCURRENCY = 12

const HTTP_JSON_AGENT = new http.Agent({ keepAlive: true, maxSockets: MAX_JSON_CONCURRENCY })
const HTTPS_JSON_AGENT = new https.Agent({ keepAlive: true, maxSockets: MAX_JSON_CONCURRENCY })

// Download pool: few long-lived streaming bodies. Isolated from the JSON pool
// so saturating it with large file downloads can't block interactive calls.
const HTTP_DOWNLOAD_AGENT = new http.Agent({ keepAlive: true, maxSockets: 8 })
const HTTPS_DOWNLOAD_AGENT = new https.Agent({ keepAlive: true, maxSockets: 8 })

function jsonAgentFor(protocol) {
  return protocol === 'https:' ? HTTPS_JSON_AGENT : HTTP_JSON_AGENT
}

// A keep-alive pool can hold a socket whose peer (an SSH mux forward channel)
// wedged mid-session: the socket stays open locally but never delivers, so the
// next reuse dies with 'socket hang up' — a false 'backend is dead' signal
// (#131765). Probes that must distinguish "stale pooled socket" from "backend
// actually gone" need a socket that CANNOT be a stale pool entry: a fresh
// agent with no keep-alive, so the TCP handshake itself proves liveness. This
// is deliberately un-pooled — callers use it for one-shot health probes only,
// never for bulk traffic.
function freshJsonAgentFor(protocol) {
  return protocol === 'https:'
    ? new https.Agent({ keepAlive: false, maxSockets: 1 })
    : new http.Agent({ keepAlive: false, maxSockets: 1 })
}

function downloadAgentFor(protocol) {
  return protocol === 'https:' ? HTTPS_DOWNLOAD_AGENT : HTTP_DOWNLOAD_AGENT
}

// Close pooled sockets so lingering keep-alive connections can't hold the
// process open (or leak FDs) across quit. Wired to app 'will-quit' in main.ts.
function destroyKeepaliveAgents() {
  for (const agent of [HTTP_JSON_AGENT, HTTPS_JSON_AGENT, HTTP_DOWNLOAD_AGENT, HTTPS_DOWNLOAD_AGENT]) {
    agent.destroy()
  }
}

// Transient transport errors: retry MAY be safe (subject to verb gating).
const TRANSIENT_CODES = new Set([
  'ECONNRESET',
  'ECONNREFUSED',
  'EPIPE',
  'ETIMEDOUT',
  'EAI_AGAIN',
  'ENOTFOUND',
  'EHOSTUNREACH',
  'ENETUNREACH'
])

// Errors that prove the request never reached the server: the TCP connection
// (or name resolution) failed outright, so nothing was submitted.
const NEVER_SENT_CODES = new Set(['ECONNREFUSED', 'ENOTFOUND', 'EAI_AGAIN', 'EHOSTUNREACH', 'ENETUNREACH'])

const IDEMPOTENT_METHODS = new Set(['GET', 'HEAD', 'OPTIONS'])

function isIdempotentMethod(method) {
  return IDEMPOTENT_METHODS.has(String(method || 'GET').toUpperCase())
}

function isTransientTransportError(error) {
  if (!error) {
    return false
  }

  if (TRANSIENT_CODES.has(error.code)) {
    return true
  }

  const msg = String(error.message || '')

  return msg.includes('socket hang up') || msg.includes('read ECONNRESET')
}

/**
 * The verb-gated retry decision.
 *
 * @param error        the transport error from the failed attempt
 * @param method       HTTP verb of the request ('GET', 'POST', ...)
 * @param requestState per-attempt state; requestState.bodySent is set true by
 *                     the caller just BEFORE the first byte of the request is
 *                     flushed, so a `false` here proves nothing went out.
 */
function shouldRetryRequest(error, method, requestState: any = {}) {
  if (!isTransientTransportError(error)) {
    return false
  }

  if (isIdempotentMethod(method)) {
    return true
  }

  // Non-idempotent: only when the request provably never reached the server.
  if (NEVER_SENT_CODES.has(error && error.code)) {
    return true
  }

  if (requestState.bodySent === false) {
    return true
  }

  // Ambiguous (reset/hang-up after the body was flushed): the server may have
  // processed it. Surface the error rather than risk a double submit.
  return false
}

/**
 * Run `makeAttempt` with bounded retries under the policy above.
 *
 * `makeAttempt(requestState)` must return a Promise and should set
 * `requestState.bodySent = true` immediately before flushing the request
 * (before the first req.write()/req.end()). Each attempt gets a fresh state
 * object initialized to { bodySent: false }.
 */
async function withRetry(makeAttempt, options: any = {}) {
  const method = String(options.method || 'GET').toUpperCase()
  const maxRetries = Number.isInteger(options.maxRetries) ? options.maxRetries : 2

  // Retry delay must be proportional to the timeout that triggered the retry,
  // not a fixed small constant. When a request times out after Nms of queueing,
  // re-entering the queue 200ms later guarantees another timeout: the queue has
  // not drained. Delay = timeout * 0.1 * 2^attempt, capped at 5s, plus +/-50%
  // jitter so concurrent failures do not retry in lockstep (which would
  // synchronise the storm instead of spreading it). For fast-fail errors
  // (ECONNRESET) where no timeout fired, falls back to the legacy 200ms base.
  const delayFn =
    options.delayFn ||
    (attempt => {
      const timeoutMs = Number(options.timeoutMs) > 0 ? Number(options.timeoutMs) : 0
      const base = timeoutMs > 0 ? timeoutMs * 0.1 : 200
      const backoff = Math.min(base * Math.pow(2, attempt), 5000)
      const jitter = backoff * 0.5 * Math.random()
      return new Promise(r => setTimeout(r, backoff + jitter))
    })

  let lastError

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    const requestState = { bodySent: false }

    try {
      return await makeAttempt(requestState)
    } catch (error) {
      lastError = error

      if (attempt < maxRetries && shouldRetryRequest(error, method, requestState)) {
        await delayFn(attempt)

        continue
      }

      throw error
    }
  }

  throw lastError
}

/**
 * The one error shape the REST helpers throw for an HTTP >= 400 response.
 *
 * `statusCode` is the structured contract every downstream classifier reads —
 * isGatewayAuthRejection (401/403 → reauth, never retried), isServerSideHttpError
 * (502/503/504 → Cloud-down), ensureNativeAccessToken's dead-refresh-token
 * check — and the "<status>: <body>" message keeps the legacy prefix readers
 * working. fetchJson used to build a bare Error here, so a native-bearer 401
 * reached the boot path as an anonymous transport failure: it was retried,
 * then classified as transient, and the renderer's boot-retry loop flickered
 * the Sign in overlay away (#95701). Shared by fetchJson, fetchPublicJson and
 * the OAuth-session fetch so the three paths cannot drift apart again.
 */
function httpStatusError(statusCode, text, statusMessage?) {
  const status = Number.isInteger(statusCode) && statusCode > 0 ? statusCode : 500
  const detail = String(text || statusMessage || '')
  const error: any = new Error(`${status}: ${detail}`)
  error.statusCode = status

  return error
}

/** Read side of httpStatusError: the HTTP status an error carries, NaN when it carries none. */
function readStatusCode(error: unknown): number {
  return Number(error && typeof error === 'object' ? (error as { statusCode?: unknown }).statusCode : NaN)
}

/**
 * The structured JSON body an httpStatusError carries after its "<status>: "
 * prefix, or null when the body was not a JSON object. NAS answers
 * `{ error: "<code>", ... }` on 4xx, and every reader of that code (the 409
 * org picker, the stale-team fallback) must parse the prefix the same way.
 */
function readJsonErrorBody(error: unknown): null | Record<string, unknown> {
  const message = error instanceof Error ? error.message : ''
  const start = message.indexOf('{')

  if (start < 0) {
    return null
  }

  try {
    const parsed: unknown = JSON.parse(message.slice(start))

    return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? (parsed as Record<string, unknown>) : null
  } catch {
    return null
  }
}

/**
 * Error for a JSON endpoint that did not answer JSON. A 2xx HTML body is the
 * SPA index.html for an unregistered /api path, and downstream capability
 * probes (isMissingHealthEndpointError, gateway-rpc) key on the "endpoint is
 * likely missing" wording. A 3xx (callers never follow redirects, whatever
 * the body) is the request being bounced elsewhere: an access proxy sending
 * it to its login page, or a scheme / trailing-slash redirect when the saved
 * URL is off by that much — so it must neither carry that wording nor blame
 * the backend, and it names the Location when the server sent one.
 */
function htmlResponseError(url: string, statusCode: unknown, location?: unknown) {
  const status = Number(statusCode)

  if (status >= 300 && status < 400) {
    const target = typeof location === 'string' && location.trim() ? location.trim() : null

    const hint = isSchemeOrSlashRedirect(url, target)
      ? 'The saved gateway URL differs from the server by scheme or trailing slash; update it to match.'
      : 'This is usually an authentication proxy in front of the gateway; check the saved token and extra gateway headers.'

    return new Error(
      `Expected JSON from ${url} but the request was redirected (status ${statusCode})${target ? ` to ${target}` : ''}. ${hint}`
    )
  }

  return new Error(
    `Expected JSON from ${url} but got HTML (status ${statusCode}). The endpoint is likely missing on the Hermes backend.`
  )
}

function isSchemeOrSlashRedirect(requestUrl: string, location: null | string): boolean {
  if (!location) {
    return false
  }

  try {
    const from = new URL(requestUrl)
    const to = new URL(location, requestUrl)
    const strip = (value: string) => value.replace(/\/+$/, '')

    return (
      from.host === to.host &&
      from.search === to.search &&
      (from.protocol !== to.protocol || from.pathname !== to.pathname) &&
      strip(from.pathname) === strip(to.pathname)
    )
  } catch {
    return false
  }
}

export {
  destroyKeepaliveAgents,
  downloadAgentFor,
  freshJsonAgentFor,
  htmlResponseError,
  httpStatusError,
  isIdempotentMethod,
  isTransientTransportError,
  jsonAgentFor,
  readJsonErrorBody,
  readStatusCode,
  shouldRetryRequest,
  withRetry
}
