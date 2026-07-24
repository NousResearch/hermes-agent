/**
 * connection-config.ts
 *
 * Pure, electron-free helpers for the desktop's remote-gateway connection
 * config: URL normalization, WS-URL construction (token vs OAuth ticket),
 * auth-mode classification, and the auth-mode coercion rules.
 *
 * Kept standalone (no `import 'electron'`) so it can be unit-tested with
 * `node --test` — same pattern as backend-probes.ts / bootstrap-platform.ts.
 * main.ts requires these and wires them into the electron-coupled IPC layer.
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

// The Nous portal (NAS) does NOT use Hermes gateway session cookies — it is a
// Privy-authed Next.js app. NAS `auth()` (src/server/auth/session.ts) reads the
// `privy-token` access-token cookie (with `privy-id-token` alongside), which is
// also exactly what the `/api/agents` cookie-auth path validates. So portal
// sign-in / discovery liveness must look for the Privy cookie, NOT the gateway
// cookies above. `privy-token` is the access token (the required signal);
// variants cover the secured-prefix forms and the older `privy-session` name.
const PRIVY_SESSION_COOKIE_VARIANTS = ['__Host-privy-token', '__Secure-privy-token', 'privy-token', 'privy-session']

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

/** True only when a gateway explicitly rejected the current OAuth session. */
function isGatewayAuthRejection(error) {
  if (error && typeof error === 'object' && (error as any).needsOauthLogin === true) {
    return true
  }

  const statusCode = Number(error && typeof error === 'object' ? (error as any).statusCode : NaN)

  return statusCode === 401 || statusCode === 403
}

function gatewayTicketFailure(error, authMessage, transportMessage) {
  const needsOauthLogin = isGatewayAuthRejection(error)
  const err = new Error(needsOauthLogin ? authMessage : transportMessage)

  if (needsOauthLogin) {
    ;(err as any).needsOauthLogin = true
  }

  err.cause = error

  return err
}

/** Serialize a fresh-WS-URL attempt across Electron's IPC boundary. */
async function gatewayWsUrlIpcResult(resolveWsUrl: () => Promise<string>) {
  try {
    return { ok: true as const, wsUrl: await resolveWsUrl() }
  } catch (error) {
    return {
      error: error instanceof Error ? error.message : String(error),
      ...(isGatewayAuthRejection(error) ? { needsOauthLogin: true as const } : {}),
      ok: false as const
    }
  }
}

/**
 * Build the WS URL the renderer would connect with, so the connection test can
 * exercise the same transport the app actually uses.
 *
 * The OAuth ticket-minter is injected (`mintTicket(baseUrl) -> Promise<ticket>`)
 * so this stays electron-free and unit-testable; main.ts passes the real
 * `mintGatewayWsTicket`.
 *
 * Return semantics:
 *   - token mode + token   → ws(s)://…/api/ws?token=…
 *   - token mode, no token → null  (genuine skip; nothing to authenticate with)
 *   - oauth, mint ok       → ws(s)://…/api/ws?ticket=…
 *   - oauth, mint fails    → THROWS  (NOT a skip)
 *
 * The oauth-mint-failure throw is the important case: swallowing it here would
 * re-introduce the exact false-positive this test exists to catch. An explicit
 * 401/403 asks for sign-in; transport and server failures remain connectivity
 * errors so a temporary outage is not mislabeled as an expired session.
 *
 * @param {string} baseUrl
 * @param {'token'|'oauth'} authMode
 * @param {string|null} token
 * @param {{ mintTicket: (baseUrl: string) => Promise<string> }} deps
 * @returns {Promise<string|null>}
 */
async function resolveTestWsUrl(baseUrl, authMode, token, deps: any = {}) {
  if (authMode === 'oauth') {
    const mintTicket = deps.mintTicket

    if (typeof mintTicket !== 'function') {
      throw new Error('resolveTestWsUrl: a mintTicket function is required in OAuth mode.')
    }

    let ticket

    try {
      ticket = await mintTicket(baseUrl)
    } catch (error) {
      throw gatewayTicketFailure(
        error,
        'Reached the gateway over HTTP, but the OAuth session was rejected while minting a WebSocket ticket. ' +
          'Open Settings → Gateway and sign in again.',
        'Reached the gateway over HTTP, but could not mint a WebSocket ticket. Check the remote gateway connection and try again.'
      )
    }

    return buildGatewayWsUrlWithTicket(baseUrl, ticket)
  }

  if (!token) {
    return null
  }

  return buildGatewayWsUrl(baseUrl, token)
}

// Normalize a profile name to a connection scope key, or null for the global
// (default) connection. Shared by the resolver and the IPC layer.
function connectionScopeKey(profile) {
  return String(profile ?? '').trim() || null
}

// Coerce a remote auth mode to one of the two supported values ('token' default).
function normAuthMode(mode) {
  return mode === 'oauth' ? 'oauth' : 'token'
}

// True for connection modes that resolve to a REMOTE backend. 'cloud' is a
// Hermes Cloud connection (cloud-auto-discovery Q3/Q6): it carries a
// remote-shaped block and reuses the entire remote connect/probe/reconnect
// path, so every resolution site treats it exactly like 'remote'. The only
// places that distinguish cloud from remote are the settings UI (which card to
// show) and config persistence (remembering the provenance). Centralized here
// so no resolution site forgets the third arm.
function modeIsRemoteLike(mode) {
  return mode === 'remote' || mode === 'cloud'
}

function normalizeSshConfig(entry) {
  if (!entry || typeof entry !== 'object' || entry.mode !== 'ssh') {
    return null
  }

  let host = String(entry.host || '').trim()

  if (!host) {
    return null
  }

  let parsedUser
  let parsedPort
  const at = host.indexOf('@')

  if (at > 0) {
    parsedUser = host.slice(0, at)
    host = host.slice(at + 1)
  }

  const bracketed = /^\[([^\]]+)](?::(\d+))?$/.exec(host)

  if (bracketed) {
    host = bracketed[1]

    if (bracketed[2]) {
      parsedPort = Number(bracketed[2])
    }
  } else if ((host.match(/:/g) || []).length === 1) {
    const [name, rawPort] = host.split(':')

    if (/^\d+$/.test(rawPort)) {
      host = name
      parsedPort = Number(rawPort)
    }
  }

  if (!host) {
    return null
  }

  const out: any = { mode: 'ssh', host }
  const user = String(entry.user || '').trim() || parsedUser || ''

  if (user) {
    out.user = user
  }

  const rawExplicitPort = String(entry.port ?? '').trim()
  const explicitPort = /^\d+$/.test(rawExplicitPort) ? Number(rawExplicitPort) : null
  const port = explicitPort ?? parsedPort

  if (Number.isInteger(port) && port > 0 && port <= 65535 && port !== 22) {
    out.port = port
  }

  const keyPath = String(entry.keyPath || '').trim()

  if (keyPath) {
    out.keyPath = keyPath
  }

  const remoteHermesPath = String(entry.remoteHermesPath || '').trim()

  if (remoteHermesPath) {
    out.remoteHermesPath = remoteHermesPath
  }

  return out
}

function profileSshOverride(config, profile) {
  const key = connectionScopeKey(profile)
  const entry = key ? config?.profiles?.[key] : null

  return normalizeSshConfig(entry)
}

function savedProfileSsh(config, profile) {
  const key = connectionScopeKey(profile)
  const entry = key ? config?.profiles?.[key] : null

  if (!entry || entry.mode !== 'local') {
    return null
  }

  return normalizeSshConfig(entry.savedSsh)
}

function profileHasRemoteConnection(config, profile) {
  return Boolean(profileRemoteOverride(config, profile) || profileSshOverride(config, profile))
}

function localProfileEntry(existing) {
  const ssh = normalizeSshConfig(existing) || normalizeSshConfig(existing?.savedSsh)

  return ssh ? { mode: 'local', savedSsh: ssh } : null
}

function hostLabelFromBaseUrl(baseUrl) {
  const raw = String(baseUrl || '').trim()

  if (!raw) {
    return null
  }

  try {
    const parsed = new URL(raw)

    if (!parsed.hostname) {
      return null
    }

    return parsed.port && parsed.port !== '80' && parsed.port !== '443'
      ? `${parsed.hostname}:${parsed.port}`
      : parsed.hostname
  } catch {
    return null
  }
}

/**
 * Select a profile's explicit remote override from a connection config, or null
 * when it has none (so the caller falls back to env → global remote → local).
 *
 * The config may carry a `profiles` map keyed by name; an entry counts as an
 * override only with a remote-like `mode` (remote or cloud) and a non-empty
 * `url`. Pure: `token` is the raw stored secret; main.ts decrypts it. Returns
 * `{ url, authMode, token } | null`.
 */
function profileRemoteOverride(config, profile) {
  const key = connectionScopeKey(profile)
  const entry = key ? config?.profiles?.[key] : null

  if (!entry || typeof entry !== 'object' || !modeIsRemoteLike(entry.mode)) {
    return null
  }

  const url = String(entry.url || '').trim()

  if (!url) {
    return null
  }

  return { url, authMode: normAuthMode(entry.authMode), token: entry.token }
}

/**
 * In global-remote mode one backend serves every Desktop profile, so REST calls
 * that are scoped by renderer-side `request.profile` must carry that scope as a
 * query parameter. Local pooled backends and per-profile remote overrides do not
 * need this: they already run against a backend scoped to the target profile.
 */
function pathWithGlobalRemoteProfile(path, profile, opts: any = {}) {
  const scopedProfile = connectionScopeKey(profile)

  if (!scopedProfile || !opts.globalRemote || opts.profileRemoteOverride) {
    return path
  }

  const rawPath = String(path || '')

  if (!rawPath) {
    return path
  }

  let parsed

  try {
    parsed = new URL(rawPath, 'http://hermes.local')
  } catch {
    return path
  }

  if (parsed.searchParams.has('profile')) {
    return path
  }

  parsed.searchParams.set('profile', scopedProfile)

  return `${parsed.pathname}${parsed.search}${parsed.hash}`
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
  if (inputAuthMode === 'oauth') {
    return 'oauth'
  }

  if (inputAuthMode === 'token') {
    return 'token'
  }

  if (existingAuthMode === 'oauth') {
    return 'oauth'
  }

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
  if (!Array.isArray(cookies)) {
    return false
  }

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
  if (!Array.isArray(cookies)) {
    return false
  }

  return cookies.some(c => c && c.value && (AT_COOKIE_VARIANTS.includes(c.name) || RT_COOKIE_VARIANTS.includes(c.name)))
}

/**
 * True if the cookie jar holds a live Nous PORTAL (Privy) session — a non-empty
 * `privy-token` (access-token) cookie, or a variant. This is the portal
 * analogue of `cookiesHaveLiveSession`: the portal authenticates via Privy, not
 * the Hermes gateway session cookies, so cloud sign-in / discovery liveness
 * must check THIS, not the gateway helpers. (NAS `auth()` and the `/api/agents`
 * cookie path both key off `privy-token`.)
 */
function cookiesHavePrivySession(cookies) {
  if (!Array.isArray(cookies)) {
    return false
  }

  return cookies.some(c => c && c.value && PRIVY_SESSION_COOKIE_VARIANTS.includes(c.name))
}

export {
  AT_COOKIE_VARIANTS,
  authModeFromStatus,
  buildGatewayWsUrl,
  buildGatewayWsUrlWithTicket,
  connectionScopeKey,
  cookiesHaveLiveSession,
  cookiesHavePrivySession,
  cookiesHaveSession,
  gatewayTicketFailure,
  gatewayWsUrlIpcResult,
  hostLabelFromBaseUrl,
  isGatewayAuthRejection,
  localProfileEntry,
  modeIsRemoteLike,
  normalizeRemoteBaseUrl,
  normalizeSshConfig,
  normAuthMode,
  pathWithGlobalRemoteProfile,
  PRIVY_SESSION_COOKIE_VARIANTS,
  profileHasRemoteConnection,
  profileRemoteOverride,
  profileSshOverride,
  resolveAuthMode,
  resolveTestWsUrl,
  RT_COOKIE_VARIANTS,
  savedProfileSsh,
  tokenPreview
}
