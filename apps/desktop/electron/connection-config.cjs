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

// Bare + prefixed variants of the access-token cookie the gateway may set,
// depending on its deploy shape (HTTPS direct → __Host-, behind a path prefix
// → __Secure-, loopback HTTP → bare). Mirrors
// hermes_cli/dashboard_auth/cookies.py.
const AT_COOKIE_VARIANTS = ['__Host-hermes_session_at', '__Secure-hermes_session_at', 'hermes_session_at']

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
 * True if any cookie in `cookies` is a hermes session access-token cookie
 * with a non-empty value. `cookies` is an array of {name, value} (the shape
 * Electron's session.cookies.get returns).
 */
function cookiesHaveSession(cookies) {
  if (!Array.isArray(cookies)) return false
  return cookies.some(c => c && AT_COOKIE_VARIANTS.includes(c.name) && c.value)
}

module.exports = {
  AT_COOKIE_VARIANTS,
  authModeFromStatus,
  buildGatewayWsUrl,
  buildGatewayWsUrlWithTicket,
  cookiesHaveSession,
  normalizeRemoteBaseUrl,
  resolveAuthMode,
  tokenPreview
}
