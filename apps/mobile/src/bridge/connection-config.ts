/**
 * connection-config.ts
 *
 * Pure, dependency-free helpers ported from the desktop's
 * `apps/desktop/electron/connection-config.cjs`. URL normalization, WS-URL
 * construction (token vs OAuth ticket), and auth-mode classification.
 *
 * IMPORTANT — basic auth is NOT a third mode. The gateway advertises
 * `auth_required: true` whenever a login gate is engaged, whether that gate is
 * an OAuth redirect or a username/password form. The cookie + ws-ticket
 * machinery is identical for both; only the login *form* differs. So we
 * classify password-gated gateways as `'oauth'` here and reuse the entire
 * ticket-minting / re-auth path unchanged.
 */

export type AuthMode = 'oauth' | 'token'

/** Cookie name variants the gateway may set depending on deploy shape. */
export const AT_COOKIE_VARIANTS = [
  '__Host-hermes_session_at',
  '__Secure-hermes_session_at',
  'hermes_session_at',
]
export const RT_COOKIE_VARIANTS = [
  '__Host-hermes_session_rt',
  '__Secure-hermes_session_rt',
  'hermes_session_rt',
]

/** Normalize a user-entered gateway URL: require http(s), strip hash/query and
 *  trailing slashes. Throws on invalid input (surfaced on the connect screen). */
export function normalizeRemoteBaseUrl(rawUrl: string): string {
  const value = String(rawUrl || '').trim()
  if (!value) throw new Error('Gateway URL is required.')

  let parsed: URL
  try {
    parsed = new URL(value)
  } catch (error) {
    throw new Error(`Gateway URL is not valid: ${(error as Error).message}`)
  }

  if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
    throw new Error(`Gateway URL must be http:// or https://, got ${parsed.protocol}`)
  }

  parsed.hash = ''
  parsed.search = ''
  parsed.pathname = parsed.pathname.replace(/\/+$/, '')

  return parsed.toString().replace(/\/+$/, '')
}

/** ws(s)://host[/prefix]/api/ws?token=… — legacy/static-token gateways. */
export function buildGatewayWsUrl(baseUrl: string, token: string): string {
  const parsed = new URL(baseUrl)
  const wsScheme = parsed.protocol === 'https:' ? 'wss' : 'ws'
  const prefix = parsed.pathname.replace(/\/+$/, '')
  return `${wsScheme}://${parsed.host}${prefix}/api/ws?token=${encodeURIComponent(token)}`
}

/** ws(s)://host[/prefix]/api/ws?ticket=… — OAuth/password-gated gateways. */
export function buildGatewayWsUrlWithTicket(baseUrl: string, ticket: string): string {
  const parsed = new URL(baseUrl)
  const wsScheme = parsed.protocol === 'https:' ? 'wss' : 'ws'
  const prefix = parsed.pathname.replace(/\/+$/, '')
  return `${wsScheme}://${parsed.host}${prefix}/api/ws?ticket=${encodeURIComponent(ticket)}`
}

/** Build an absolute REST URL for a gateway path. */
export function buildApiUrl(baseUrl: string, path: string): string {
  const parsed = new URL(baseUrl)
  const prefix = parsed.pathname.replace(/\/+$/, '')
  const rel = path.startsWith('/') ? path : `/${path}`
  return `${parsed.protocol}//${parsed.host}${prefix}${rel}`
}

/** Classify auth mode from a gateway's public /api/status body.
 *  `auth_required: true` → a login gate (oauth OR password) → 'oauth'. */
export function authModeFromStatus(statusBody: unknown): AuthMode {
  return statusBody && (statusBody as { auth_required?: boolean }).auth_required
    ? 'oauth'
    : 'token'
}
