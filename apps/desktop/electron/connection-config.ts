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

// Saved gateway connections are deliberately separate from Hermes Agent
// profiles. A connection is a machine/backend endpoint; a Hermes profile is a
// config/session namespace served by that endpoint. Conflating the two was the
// old workaround for storing more than one remote host.
const SAVED_CONNECTION_ID_RE = /^[a-z0-9][a-z0-9_-]{0,63}$/

function savedConnectionIdFromName(name) {
  const raw = String(name || '')
    .trim()
    .toLowerCase()

  if (!raw) {
    throw new Error('Connection name is required.')
  }

  if (raw === 'local') {
    throw new Error('Connection name cannot be “local”.')
  }

  let id = raw
    .replace(/[^a-z0-9_-]+/g, '-')
    .replace(/^[^a-z0-9]+|[^a-z0-9]+$/g, '')
    .slice(0, 64)

  // Display names may be entirely non-Latin. Keep the shortcut shell-safe and
  // stable by deriving a short deterministic fallback rather than rejecting a
  // perfectly valid localized name.
  if (!id) {
    let hash = 2166136261

    for (let index = 0; index < raw.length; index += 1) {
      hash ^= raw.charCodeAt(index)
      hash = Math.imul(hash, 16777619)
    }

    id = `connection-${(hash >>> 0).toString(36)}`
  }

  if (!SAVED_CONNECTION_ID_RE.test(id)) {
    throw new Error('Connection name must contain at least one letter or number.')
  }

  return id
}

function sanitizeSavedConnections(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return {}
  }

  const out: Record<string, any> = {}

  for (const [rawId, rawEntry] of Object.entries(value)) {
    const id = String(rawId || '')
      .trim()
      .toLowerCase()

    if (!SAVED_CONNECTION_ID_RE.test(id) || id === 'local' || !rawEntry || typeof rawEntry !== 'object') {
      continue
    }

    const entry: any = rawEntry

    const name = String(entry.name || id)
      .trim()
      .slice(0, 80)

    const url = String(entry.url || '').trim()

    if (!name || !url) {
      continue
    }

    out[id] = {
      name,
      url,
      authMode: normAuthMode(entry.authMode),
      token: entry.token
    }
  }

  return out
}

function savedConnectionEntries(config) {
  const connections = sanitizeSavedConnections(config?.connections)

  return Object.entries(connections)
    .map(([id, entry]: [string, any]) => ({ id, ...entry }))
    .sort((a, b) => a.name.localeCompare(b.name))
}

function resolveSavedConnection(config, selector) {
  const value = String(selector || '').trim()

  if (!value) {
    return null
  }

  if (value.toLowerCase() === 'local') {
    return { id: 'local', name: 'Local gateway', local: true }
  }

  const entries = savedConnectionEntries(config)
  const byId = entries.find(entry => entry.id === value.toLowerCase())

  if (byId) {
    return byId
  }

  const folded = value.toLocaleLowerCase()
  const byName = entries.filter(entry => entry.name.toLocaleLowerCase() === folded)

  return byName.length === 1 ? byName[0] : null
}

function selectSavedConnection(config, selector) {
  const resolved: any = resolveSavedConnection(config, selector)

  if (!resolved) {
    const available = savedConnectionEntries(config).map(entry => entry.id)

    const suffix = available.length
      ? ` Available connections: ${available.join(', ')}.`
      : ' No remote connections are saved.'

    throw new Error(`Unknown desktop connection “${String(selector || '').trim()}”.${suffix}`)
  }

  if (resolved.local) {
    return { ...config, mode: 'local' }
  }

  const { id, name: _name, ...remote } = resolved

  return {
    ...config,
    mode: 'remote',
    selectedConnection: id,
    remote
  }
}

function removeSavedConnection(config, selector) {
  const resolved: any = resolveSavedConnection(config, selector)

  if (!resolved || resolved.local) {
    throw new Error(`Unknown remote desktop connection “${String(selector || '').trim()}”.`)
  }

  const connections = { ...sanitizeSavedConnections(config?.connections) }
  delete connections[resolved.id]

  const selectedConnection = config?.selectedConnection === resolved.id ? null : config?.selectedConnection || null
  const mode = config?.selectedConnection === resolved.id && config?.mode === 'remote' ? 'local' : config?.mode

  return { ...config, connections, selectedConnection, mode }
}

function parseDesktopConnectionArg(argv) {
  if (!Array.isArray(argv)) {
    return null
  }

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index]

    if (typeof arg !== 'string') {
      continue
    }

    if (arg === '--connection') {
      const next = argv[index + 1]

      return typeof next === 'string' && !next.startsWith('--') ? next.trim() || null : null
    }

    if (arg.startsWith('--connection=')) {
      return arg.slice('--connection='.length).trim() || null
    }
  }

  return null
}

function normalizeStoredConnectionConfig(parsed) {
  const value = parsed && typeof parsed === 'object' ? parsed : {}
  let remote = value.remote && typeof value.remote === 'object' ? { ...value.remote } : {}
  remote.authMode = normAuthMode(remote.authMode)

  let connections = sanitizeSavedConnections(value.connections)

  const parsedSelection =
    typeof value.selectedConnection === 'string' ? value.selectedConnection.trim().toLowerCase() : ''

  let selectedConnection = parsedSelection && connections[parsedSelection] ? parsedSelection : null

  // Backward migration: the old schema retained one remote block even while
  // Local was selected. Lift it into the registry without turning Local on.
  // A Cloud block belongs to discovery, not to the user's named remotes.
  if (
    Object.keys(connections).length === 0 &&
    value.mode !== 'cloud' &&
    typeof remote.url === 'string' &&
    remote.url.trim()
  ) {
    connections = {
      remote: {
        name: 'Remote gateway',
        url: remote.url,
        authMode: normAuthMode(remote.authMode),
        token: remote.token
      }
    }
    selectedConnection = 'remote'
  }

  if (value.mode === 'remote' && selectedConnection && connections[selectedConnection]) {
    const { name: _name, ...selectedRemote } = connections[selectedConnection]
    remote = selectedRemote
  }

  return {
    mode: modeIsRemoteLike(value.mode) ? value.mode : 'local',
    remote,
    connections,
    selectedConnection
  }
}

function upsertSavedConnection(config, input) {
  const connections = sanitizeSavedConnections(config?.connections)

  const storedSelection =
    typeof config?.selectedConnection === 'string' && connections[config.selectedConnection]
      ? config.selectedConnection
      : null

  const explicitConnectionId = typeof input?.connectionId === 'string' ? input.connectionId.trim().toLowerCase() : null

  if (explicitConnectionId && !connections[explicitConnectionId]) {
    throw new Error(`Unknown saved connection “${explicitConnectionId}”.`)
  }

  const selectedConnection =
    explicitConnectionId && connections[explicitConnectionId] ? explicitConnectionId : storedSelection

  const selectedBlock = selectedConnection ? connections[selectedConnection] : null
  const forceNew = Object.prototype.hasOwnProperty.call(input || {}, 'connectionId') && input.connectionId === null
  const requestedName = String(input?.connectionName || selectedBlock?.name || 'Remote gateway').trim()

  if (!requestedName) {
    throw new Error('Connection name is required.')
  }

  let id = forceNew ? savedConnectionIdFromName(requestedName) : explicitConnectionId || storedSelection

  if (!id) {
    id = savedConnectionIdFromName(requestedName)
  }

  if (forceNew && connections[id]) {
    throw new Error(`A saved connection with shortcut name “${id}” already exists.`)
  }

  return {
    connections: {
      ...connections,
      [id]: {
        name: requestedName.slice(0, 80),
        url: input.remote.url,
        authMode: normAuthMode(input.remote.authMode),
        token: input.remote.token
      }
    },
    selectedConnection: id
  }
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
  isGatewayAuthRejection,
  modeIsRemoteLike,
  normalizeRemoteBaseUrl,
  normalizeStoredConnectionConfig,
  normAuthMode,
  parseDesktopConnectionArg,
  pathWithGlobalRemoteProfile,
  PRIVY_SESSION_COOKIE_VARIANTS,
  profileRemoteOverride,
  removeSavedConnection,
  resolveAuthMode,
  resolveSavedConnection,
  resolveTestWsUrl,
  RT_COOKIE_VARIANTS,
  sanitizeSavedConnections,
  savedConnectionEntries,
  savedConnectionIdFromName,
  selectSavedConnection,
  tokenPreview,
  upsertSavedConnection
}
