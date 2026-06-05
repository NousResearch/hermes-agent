/**
 * connection-config.cjs
 *
 * Pure, electron-free helpers for the desktop's remote-gateway connection
 * config: URL normalization, WS-URL construction (token vs OAuth ticket),
 * auth-mode classification, and the auth-mode coercion rules.
 *
 * Kept standalone (no `require('electron')`) so it can be unit-tested with
 * `node --test` — same pattern as backend-probes.cjs / bootstrap-platform.cjs.
 * main.cjs requires these and wires them into the electron-coupled IPC layer.
 *
 * Background on the two auth models a remote gateway can use:
 *   - 'token': legacy static dashboard session token. REST uses an
 *     `X-Hermes-Session-Token` header; WS uses `?token=`.
 *   - 'oauth': hosted gateways gate behind an OAuth provider. REST is authed
 *     by an HttpOnly session cookie; WS upgrades require a single-use
 *     `?ticket=` minted at POST /api/auth/ws-ticket. The gateway advertises
 *     this via the public `/api/status` field `auth_required: true`.
 */

// Bare + prefixed variants of the session cookies the gateway may set,
// depending on its deploy shape (HTTPS direct → __Host-, behind a path prefix
// → __Secure-, loopback HTTP → bare). Mirrors
// hermes_cli/dashboard_auth/cookies.py.
//
// Two cookies are in play (see that module):
//   - hermes_session_at: the OAuth access token. Short-lived (~15 min); its
//     Max-Age tracks the access-token TTL, so the cookie jar drops it the
//     instant the AT expires.
//   - hermes_session_rt: the OAuth refresh token. Long-lived (24h rotating,
//     reuse-detected — Portal NAS #293 / hermes #37247). When the AT cookie
//     has lapsed but the RT cookie is still present, the gateway middleware
//     transparently rotates a fresh AT on the next authenticated request
//     (POST /api/auth/ws-ticket), so the session is still LIVE even with no
//     AT cookie. A liveness check that looked only at the AT cookie would
//     force a needless full re-login every ~15 min — hence cookiesHaveLiveSession.
const AT_COOKIE_VARIANTS = ['__Host-hermes_session_at', '__Secure-hermes_session_at', 'hermes_session_at']
const RT_COOKIE_VARIANTS = ['__Host-hermes_session_rt', '__Secure-hermes_session_rt', 'hermes_session_rt']

function normalizeRemoteBaseUrl(rawUrl) {
  const value = String(rawUrl || '').trim()

  if (!value) {
    throw new Error('Remote gateway URL is required.')
  }

  let parsed
  try {
    parsed = new URL(value)
  } catch (error) {
    throw new Error(`Remote gateway URL is not valid: ${error.message}`)
  }

  if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
    throw new Error(`Remote gateway URL must be http:// or https://, got ${parsed.protocol}`)
  }

  parsed.hash = ''
  parsed.search = ''
  parsed.pathname = parsed.pathname.replace(/\/+$/, '')

  return parsed.toString().replace(/\/+$/, '')
}

function buildGatewayWsUrl(baseUrl, token) {
  const parsed = new URL(baseUrl)
  const wsScheme = parsed.protocol === 'https:' ? 'wss' : 'ws'
  const prefix = parsed.pathname.replace(/\/+$/, '')

  return `${wsScheme}://${parsed.host}${prefix}/api/ws?token=${encodeURIComponent(token)}`
}

function buildGatewayWsUrlWithTicket(baseUrl, ticket) {
  const parsed = new URL(baseUrl)
  const wsScheme = parsed.protocol === 'https:' ? 'wss' : 'ws'
  const prefix = parsed.pathname.replace(/\/+$/, '')

  return `${wsScheme}://${parsed.host}${prefix}/api/ws?ticket=${encodeURIComponent(ticket)}`
}

function tokenPreview(value) {
  const raw = String(value || '')

  if (!raw) {
    return null
  }

  return raw.length <= 8 ? 'set' : `...${raw.slice(-6)}`
}

/**
 * Classify a gateway's auth mode from its public /api/status body.
 * `auth_required: true` → OAuth gate engaged; otherwise legacy token auth.
 * Returns 'oauth' | 'token'.
 */
function authModeFromStatus(statusBody) {
  return statusBody && statusBody.auth_required ? 'oauth' : 'token'
}

/**
 * Resolve the effective auth mode for a coerce/save operation.
 * Explicit input wins; otherwise inherit the saved value; default 'token'.
 * Returns 'oauth' | 'token'.
 */
function resolveAuthMode(inputAuthMode, existingAuthMode) {
  if (inputAuthMode === 'oauth') return 'oauth'
  if (inputAuthMode === 'token') return 'token'
  if (existingAuthMode === 'oauth') return 'oauth'
  return 'token'
}

/**
 * True if any cookie in `cookies` is a hermes session ACCESS-token cookie
 * with a non-empty value. `cookies` is an array of {name, value} (the shape
 * Electron's session.cookies.get returns).
 *
 * Note: this is AT-only. A session whose AT cookie has lapsed but whose RT
 * cookie is still alive is STILL connectable (the gateway refreshes the AT on
 * the next request) — use `cookiesHaveLiveSession` for a connectivity/display
 * check. `cookiesHaveSession` remains exported for callers that specifically
 * need to know whether an unexpired access token is present right now.
 */
function cookiesHaveSession(cookies) {
  if (!Array.isArray(cookies)) return false
  return cookies.some(c => c && AT_COOKIE_VARIANTS.includes(c.name) && c.value)
}

/**
 * True if the cookie jar holds a credential that can yield an authenticated
 * request — EITHER a live access-token cookie OR a refresh-token cookie. The
 * RT cookie outlives the AT cookie (24h vs ~15min), and the gateway middleware
 * transparently rotates a fresh AT from the RT on the next authenticated
 * request. Gating connectivity on the AT alone would force a full IDP
 * re-login every ~15 min even though a valid 24h RT is sitting in the jar.
 *
 * This answers "should we even attempt to connect / show as signed in?", not
 * "is the access token unexpired?". The authoritative liveness check is still
 * the actual ws-ticket mint at connect time (which surfaces a true 401 when
 * the RT is also dead/revoked).
 */
function cookiesHaveLiveSession(cookies) {
  if (!Array.isArray(cookies)) return false
  return cookies.some(
    c =>
      c &&
      c.value &&
      (AT_COOKIE_VARIANTS.includes(c.name) || RT_COOKIE_VARIANTS.includes(c.name))
  )
}

module.exports = {
  AT_COOKIE_VARIANTS,
  RT_COOKIE_VARIANTS,
  authModeFromStatus,
  buildGatewayWsUrl,
  buildGatewayWsUrlWithTicket,
  cookiesHaveSession,
  cookiesHaveLiveSession,
  normalizeRemoteBaseUrl,
  resolveAuthMode,
  tokenPreview
}
