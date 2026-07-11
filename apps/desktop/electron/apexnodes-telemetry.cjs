'use strict'

/**
 * apexnodes-telemetry.cjs — hc-473 (fork half): anonymous desktop
 * install/update/apply beacon emitter.
 *
 * The cloud half (hermes-cloud PR #549, `app/routers/desktop.py`
 * `ingest_desktop_telemetry`) already ships:
 *
 *     POST https://api.apex-nodes.com/api/v1/desktop/telemetry
 *
 * (api.apex-nodes.com proxies 1:1 to the same FastAPI backend as
 * apex-nodes.com/api — deploy/nginx/api.apex-nodes.com.conf `location / {
 * proxy_pass http://127.0.0.1:8000; }` — so this is the exact same host every
 * other desktop -> scheduler call already uses, see apex-managed.cjs
 * `DEFAULT_API_BASE` / `APEXNODES_API_BASE`). Public, unauthenticated,
 * rate-limited per-IP; carries NO user identity by design (no account, email,
 * session or IP column on the `desktop_install_events` table it writes to).
 *
 * This module is the ONE place that actually calls that endpoint. The three
 * instrumentation call sites — bootstrap-runner.cjs (install-stage loop),
 * shell-updater.cjs (electron-updater lifecycle), apex-bundle-install.cjs
 * (F1 download / F2 verify / C1 switch) — `require` it directly and fire a
 * beacon inline; there is no main.cjs wiring to do, wrapping the real
 * `sendDesktopTelemetry` is the default for all three so telemetry ships the
 * moment this file is required, with zero call-site changes to main.cjs.
 *
 * Request body is a hard allow-list of exactly seven anonymous fields,
 * mirroring `app.routers.desktop.DesktopTelemetryEvent` field-for-field:
 *   platform, arch, app_version, runtime_key, stage, status, error_code
 * `buildPayload` only ever reads these — no other key on the caller's event
 * object can leak into the request body, structurally.
 *
 * Five safety guarantees (hc-473 dispatch, non-negotiable):
 *   1. fire-and-forget — callers never `await` this on the real install/update
 *      critical path (see `fireTelemetry`, the wrapper the 3 call sites use).
 *   2. 3s timeout, socket `unref()`d — a dangling beacon can never delay a
 *      response nor keep a short-lived process alive past its own exit.
 *   3. silent failure — `sendDesktopTelemetry` NEVER throws or rejects; every
 *      failure mode (offline, DNS, 4xx/5xx, malformed event) resolves to
 *      `{ok:false, ...}` instead.
 *   4. offline / undelivered = dropped, no retry (v1) — no local queue, no
 *      resend-on-reconnect. A gap in the funnel is honest signal, not a bug.
 *   5. `APEXNODES_TELEMETRY=off` is a hard, checked-first kill switch.
 *
 * error_code convention: every call site builds it as `${stage}:${category}`
 * via `buildErrorCode` / `classifyErrorCategory` — a small fixed vocabulary
 * (timeout/network/permission/not_found/checksum_mismatch/exit_nonzero/
 * protocol/incompatible/cancelled/unknown) derived from the Error's own
 * message text, NEVER the raw message itself — so a stray absolute path,
 * hostname, or username embedded in a Node error message (an ENOENT, say)
 * never reaches the anonymous beacon.
 */

const http = require('node:http')
const https = require('node:https')

// Mirrors app.models.desktop_install_event.{STATUS_START,STATUS_SUCCESS,
// STATUS_FAILURE,VALID_STATUSES} exactly — the cloud Pydantic validator
// (DesktopTelemetryEvent._status_must_be_known) 422s on anything else.
const STATUS_START = 'start'
const STATUS_SUCCESS = 'success'
const STATUS_FAILURE = 'failure'
const VALID_STATUSES = new Set([STATUS_START, STATUS_SUCCESS, STATUS_FAILURE])

// Same default host + same override env var apex-managed.cjs already uses for
// every other desktop -> scheduler call (resolveApexEndpoints). Deliberately
// reused rather than inventing a telemetry-specific env var: one override
// redirects every desktop->cloud call at once (staging, region migration,
// ops emergency), this beacon included.
const DEFAULT_API_BASE = 'https://api.apex-nodes.com'
const TELEMETRY_PATH = '/api/v1/desktop/telemetry'
const DEFAULT_TIMEOUT_MS = 3000

// Allow-list + per-field cap mirroring DesktopTelemetryEvent's Pydantic
// Field(max_length=...) exactly (platform 16, arch 16, app_version 40,
// runtime_key 80, stage 32, status handled separately via VALID_STATUSES,
// error_code 120). Enforced here too (not just trusted server-side) so a
// caller mistake (e.g. passing a raw error message as error_code instead of
// running it through classifyErrorCategory first) can never silently balloon
// the request instead of erroring loudly — it's just truncated, telemetry is
// never worth failing a build over.
const FIELD_MAX_LENGTHS = {
  platform: 16,
  arch: 16,
  app_version: 40,
  runtime_key: 80,
  stage: 32,
  status: 16,
  error_code: 120
}
const ANON_FIELDS = Object.keys(FIELD_MAX_LENGTHS)

function trimTrailingSlash(value) {
  return String(value || '').replace(/\/+$/, '')
}

/**
 * `APEXNODES_TELEMETRY=off` (also accepts 0/false/disabled) — the one-switch
 * kill switch. Checked FIRST in `sendDesktopTelemetry`, before any payload
 * work or network resolution.
 * @param {Record<string,string|undefined>} [env]
 */
function isTelemetryDisabled(env = process.env) {
  const raw = String((env && env.APEXNODES_TELEMETRY) || '')
    .trim()
    .toLowerCase()
  return raw === 'off' || raw === '0' || raw === 'false' || raw === 'disabled'
}

/** @param {Record<string,string|undefined>} [env] */
function telemetryEndpoint(env = process.env) {
  const base = trimTrailingSlash((env && env.APEXNODES_API_BASE) || DEFAULT_API_BASE)
  return `${base}${TELEMETRY_PATH}`
}

/**
 * Strip an event object down to the allow-listed anonymous fields, dropping
 * undefined/null/empty values (the cloud model treats an absent optional key
 * and an explicit null the same way) and truncating anything over its cap.
 * This is the ONLY place a field can make it into the request body — nothing
 * outside ANON_FIELDS is ever read off the caller's event object.
 */
function buildPayload(event) {
  const payload = {}
  for (const field of ANON_FIELDS) {
    const value = event ? event[field] : undefined
    if (value === undefined || value === null || value === '') continue
    const str = String(value)
    const max = FIELD_MAX_LENGTHS[field]
    payload[field] = max && str.length > max ? str.slice(0, max) : str
  }
  return payload
}

/**
 * win32/darwin -> the 'win'/'mac'/'linux' vocabulary the cloud + the bundle
 * manifest (`manifest.os`) + shell-updater's own feed-URL ternary already
 * use. Centralized here so every telemetry call site (and shell-updater,
 * which had its own inline copy of this exact ternary before hc-473) agrees
 * on the same three spellings.
 * @param {string} rawPlatform e.g. process.platform, or an already-normalized 'win'/'mac'
 */
function normalizeDesktopPlatform(rawPlatform) {
  if (rawPlatform === 'darwin') return 'mac'
  if (rawPlatform === 'win32') return 'win'
  if (rawPlatform === 'mac' || rawPlatform === 'win' || rawPlatform === 'linux') return rawPlatform
  return rawPlatform || 'linux'
}

/**
 * Categorize a raw Error/message into a small, fixed, PII-free bucket. Only
 * the bucket name is ever sent — never err.message, err.stack, or anything
 * that could carry an absolute path, hostname, or username.
 * @param {Error|string|null|undefined} err
 * @returns {string}
 */
function classifyErrorCategory(err) {
  const message = String((err && err.message) || err || '').toLowerCase()
  if (!message) return 'unknown'
  if (/cancelled|canceled|aborted/.test(message)) return 'cancelled'
  if (/timed out|timeout|etimedout/.test(message)) return 'timeout'
  if (/sha256|sha mismatch|checksum|hash mismatch/.test(message)) return 'checksum_mismatch'
  if (/enoent|not found|no such file|missing/.test(message)) return 'not_found'
  if (/eacces|eperm|permission denied/.test(message)) return 'permission'
  if (/econnrefused|econnreset|enotfound|getaddrinfo|network|socket/.test(message)) return 'network'
  if (/no json result frame|no parseable json|unexpected args/.test(message)) return 'protocol'
  if (/platform_mismatch|key_mismatch|min_desktop_version|bad_manifest/.test(message)) return 'incompatible'
  if (/exit code|exit \d|non-zero/.test(message)) return 'exit_nonzero'
  return 'unknown'
}

/**
 * `${stage}:${category}` — the error_code shape every call site uses, capped
 * to the cloud's error_code max_length (also enforced independently by
 * buildPayload, this just keeps the string sane before it gets there).
 * @param {string} stage
 * @param {Error|string|null|undefined} err
 */
function buildErrorCode(stage, err) {
  const code = `${stage}:${classifyErrorCategory(err)}`
  return code.length > FIELD_MAX_LENGTHS.error_code ? code.slice(0, FIELD_MAX_LENGTHS.error_code) : code
}

/**
 * Low-level fire-and-forget JSON POST. NEVER rejects or throws into the
 * caller — every outcome (success, 4xx/5xx, timeout, DNS failure, bad URL)
 * resolves to `{ok, status?, error?}`. No retry: a single attempt, dropped on
 * failure (v1 — see the module docstring's guarantee #4).
 *
 * Uses node:http/https (matching main.cjs's own `fetchPublicJson` +
 * bootstrap-runner.cjs's install-script downloader) rather than global
 * fetch, for one consistent desktop -> cloud transport convention.
 *
 * @param {string} url
 * @param {object} payload already allow-listed (buildPayload's output)
 * @param {{timeoutMs?: number}} [opts]
 * @returns {Promise<{ok:boolean, status?:number, error?:string}>}
 */
function postJson(url, payload, { timeoutMs = DEFAULT_TIMEOUT_MS } = {}) {
  return new Promise(resolve => {
    let parsed
    try {
      parsed = new URL(url)
    } catch (err) {
      resolve({ ok: false, error: `invalid_url: ${(err && err.message) || err}` })
      return
    }
    if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
      resolve({ ok: false, error: `unsupported_protocol: ${parsed.protocol}` })
      return
    }
    const client = parsed.protocol === 'https:' ? https : http
    const body = Buffer.from(JSON.stringify(payload || {}))

    let settled = false
    const finish = result => {
      if (settled) return
      settled = true
      resolve(result)
    }

    let req
    try {
      req = client.request(parsed, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Content-Length': String(body.length) }
      })
    } catch (err) {
      finish({ ok: false, error: (err && err.message) || 'request_failed' })
      return
    }

    // A dangling beacon must never keep a short-lived process (a future CLI
    // caller, a test runner) alive past its own natural exit — unref the
    // socket the moment one is assigned.
    req.on('socket', socket => {
      try {
        socket.unref()
      } catch {
        void 0
      }
    })
    req.on('response', res => {
      // We don't care about the response body — just drain it so the socket
      // can close cleanly, and resolve on the status code.
      res.on('data', () => {})
      res.on('end', () => finish({ ok: (res.statusCode || 0) < 400, status: res.statusCode }))
      res.on('error', () => finish({ ok: false, error: 'response_error' }))
    })
    req.on('error', err => finish({ ok: false, error: (err && err.message) || 'request_error' }))
    req.setTimeout(timeoutMs, () => {
      finish({ ok: false, error: 'timeout' })
      req.destroy(new Error(`telemetry POST timed out after ${timeoutMs}ms`))
    })
    req.end(body)
  })
}

/**
 * Send one anonymous install/update telemetry beacon (hc-473). NEVER throws
 * or rejects — every failure mode (disabled, invalid event, offline, 4xx/5xx,
 * timeout) resolves to `{ok:false, ...}` instead, so a caller can fire this
 * inline without a try/catch. Still, call sites go through `fireTelemetry`
 * below rather than depending on that guarantee alone (defense in depth
 * against a future refactor here, or a test's own fake `_post`/`sendFn`
 * misbehaving).
 *
 * @param {{platform:string, arch?:string, app_version?:string,
 *   runtime_key?:string, stage:string, status:string, error_code?:string}} event
 * @param {object} [opts]
 * @param {Record<string,string|undefined>} [opts.env] defaults to process.env
 * @param {number} [opts.timeoutMs] defaults to 3000
 * @param {(url:string, payload:object, opts?:object) => Promise<{ok:boolean}>} [opts._post]
 *   transport override — the module's own tests inject a fake here instead of
 *   ever touching the real network.
 * @returns {Promise<{ok:boolean, skipped?:string, status?:number, error?:string}>}
 */
function sendDesktopTelemetry(event, opts = {}) {
  const env = opts.env || process.env
  if (isTelemetryDisabled(env)) return Promise.resolve({ ok: false, skipped: 'disabled' })
  if (!event || typeof event.stage !== 'string' || !event.stage.trim()) {
    return Promise.resolve({ ok: false, skipped: 'invalid_stage' })
  }
  if (!VALID_STATUSES.has(event.status)) {
    return Promise.resolve({ ok: false, skipped: 'invalid_status' })
  }
  if (!event.platform || typeof event.platform !== 'string') {
    return Promise.resolve({ ok: false, skipped: 'missing_platform' })
  }

  const post = opts._post || postJson
  const payload = buildPayload(event)
  const timeoutMs = opts.timeoutMs || DEFAULT_TIMEOUT_MS

  let result
  try {
    result = post(telemetryEndpoint(env), payload, { timeoutMs })
  } catch (err) {
    return Promise.resolve({ ok: false, error: (err && err.message) || 'post_threw' })
  }
  return Promise.resolve(result).catch(err => ({ ok: false, error: (err && err.message) || 'post_rejected' }))
}

/**
 * Fire-and-forget wrapper — the shape every instrumentation call site
 * actually uses. Never `await`ed by its callers (that would make telemetry a
 * hard dependency of the install/update path, exactly what hc-473 forbids);
 * this wrapper's own job is making sure that's safe regardless of what
 * `sendFn` does — a synchronously-throwing or rejecting `sendFn` (a test
 * fake, or some future override) can still never affect the caller.
 *
 * @param {(event:object) => any} sendFn usually sendDesktopTelemetry, or a test fake
 * @param {object} event
 */
function fireTelemetry(sendFn, event) {
  try {
    const result = typeof sendFn === 'function' ? sendFn(event) : null
    if (result && typeof result.catch === 'function') result.catch(() => {})
  } catch {
    // Telemetry must never affect the real install/update flow (hc-473).
    void 0
  }
}

module.exports = {
  STATUS_START,
  STATUS_SUCCESS,
  STATUS_FAILURE,
  VALID_STATUSES,
  DEFAULT_API_BASE,
  TELEMETRY_PATH,
  DEFAULT_TIMEOUT_MS,
  FIELD_MAX_LENGTHS,
  ANON_FIELDS,
  isTelemetryDisabled,
  telemetryEndpoint,
  buildPayload,
  normalizeDesktopPlatform,
  classifyErrorCategory,
  buildErrorCode,
  postJson,
  sendDesktopTelemetry,
  fireTelemetry
}
