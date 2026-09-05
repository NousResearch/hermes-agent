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
const PRIVY_SESSION_COOKIE_VARIANTS = [
  '__Host-privy-token',
  '__Secure-privy-token',
  'privy-token',
  'privy-session',
  'privy-refresh-token'
]

// The short-lived Privy ACCESS token only — the credential `/api/agents`
// actually validates. `privy-session` / `privy-refresh-token` are long-lived
// renewal material: their presence means the session is RENEWABLE (signed in,
// no interactive login needed), but discovery still 401s until a fresh
// `privy-token` is minted. Distinguishing the two is what lets a cold start
// silently renew instead of demanding a re-login (#73495).
const PRIVY_ACCESS_COOKIE_VARIANTS = ['__Host-privy-token', '__Secure-privy-token', 'privy-token']
// Keep this aligned with hermes_cli.profiles.validate_profile_name(). `default`
// is the built-in root alias; these names cannot be created as profiles.
const RESERVED_REMOTE_PROFILES = new Set(['hermes', 'test', 'tmp', 'root', 'sudo'])

// Upper bound for a per-connection `readyTimeoutMs` override. Generous — 10
// minutes covers any legitimately glacial cold start — but finite, so a typo
// like 99999999 clamps instead of hanging the ready wait (#97264).
const READY_TIMEOUT_MS_CAP = 10 * 60_000

function normalizeRemoteBaseUrl(rawUrl) {
  let value = String(rawUrl || '').trim()

  if (!value) {
    throw new Error('Remote gateway URL is required.')
  }

  // Users routinely paste scheme-less "host:port" (a Tailscale IP, a LAN
  // hostname). Without this, `new URL('100.64.0.1:9119')` either throws or —
  // worse — parses `host:` as the protocol and produces a baffling
  // "must be http:// or https://, got myhost:" error. Only a real
  // `scheme://` prefix opts out, so explicit non-http schemes (ftp://,
  // file://) still reach the protocol check below and get rejected.
  if (!/^[a-z][a-z0-9+.-]*:\/\//i.test(value)) {
    value = `http://${value}`
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

  // Preserve structured HTTP context when the source error carried an integer
  // statusCode (the fetch layer attaches err.statusCode). Downstream Cloud
  // classification (isServerSideHttpError / makeNousCloudBackendDownError) and
  // the renderer overlay depend on it surviving the ticket-error wrapper. Auth
  // semantics are unchanged: 401/403 route to reauth, 5xx stays a transport
  // failure, everything else keeps current behavior.
  const sourceStatus = Number(error && typeof error === 'object' ? (error as any).statusCode : NaN)

  if (Number.isInteger(sourceStatus)) {
    ;(err as any).statusCode = sourceStatus
  }

  err.cause = error

  return err
}

/**
 * Retry a one-shot mint/fetch that can flap on brief network blips.
 * Auth rejections (401/403 / needsOauthLogin) fail immediately — retrying those
 * just hammers a dead session. Transport/server failures retry with short delays.
 */
async function withTransientRetries(run, options: any = {}) {
  const attempts = Number.isInteger(options.attempts) && options.attempts > 0 ? options.attempts : 3
  const delaysMs = Array.isArray(options.delaysMs) && options.delaysMs.length > 0 ? options.delaysMs : [250, 750]

  const sleep =
    typeof options.sleep === 'function'
      ? options.sleep
      : (ms: number) => new Promise(resolve => setTimeout(resolve, ms))

  const isRetryable =
    typeof options.isRetryable === 'function' ? options.isRetryable : (error: unknown) => !isGatewayAuthRejection(error)

  let lastError: unknown

  for (let attempt = 0; attempt < attempts; attempt += 1) {
    try {
      return await run()
    } catch (error) {
      lastError = error

      if (!isRetryable(error) || attempt >= attempts - 1) {
        throw error
      }

      const delay = delaysMs[Math.min(attempt, delaysMs.length - 1)]
      await sleep(delay)
    }
  }

  throw lastError
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

/** Which Hermes profile the remote SSH dashboard should actually run as.
 *  Registry pool keys (`conn:mac-mini::default`) are desktop routing labels —
 *  they must never be sent to the remote as a profile name. `default` and
 *  empty mean the remote root home. */
function resolveRemoteSshDashboardProfile(configuredRemoteProfile, poolOrProfileKey) {
  const configured = String(configuredRemoteProfile || '').trim()

  if (configured && configured !== 'default') {
    return configured
  }

  const key = String(poolOrProfileKey || '').trim()
  const requested = key.startsWith('conn:') ? key.split('::').pop() || '' : key

  if (!requested || requested === 'default') {
    return ''
  }

  return requested
}

// Coerce a remote auth mode to one of the two supported values ('token' default).
function normAuthMode(mode) {
  return mode === 'oauth' ? 'oauth' : 'token'
}

const REMOTE_HEADER_NAME_RE = /^[!#$%&'*+.^_`|~0-9A-Za-z-]+$/

const FORBIDDEN_REMOTE_HEADER_NAMES = new Set([
  'authorization',
  'connection',
  'content-length',
  'content-type',
  'cookie',
  'host',
  'origin',
  'referer',
  'te',
  'trailer',
  'transfer-encoding',
  'upgrade',
  'x-hermes-session-token'
])

function normalizeRemoteHeaders(raw) {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    return {}
  }

  const out = {}

  for (const [name, secret] of Object.entries(raw)) {
    const headerName = String(name || '').trim()
    const lower = headerName.toLowerCase()

    if (!headerName || !REMOTE_HEADER_NAME_RE.test(headerName) || FORBIDDEN_REMOTE_HEADER_NAMES.has(lower)) {
      continue
    }

    if (typeof secret === 'string') {
      const value = secret.trim()

      if (value) {
        out[headerName] = { encoding: 'plain', value }
      }

      continue
    }

    if (secret && typeof secret === 'object') {
      const encoding = String((secret as any).encoding || '')
      const value = String((secret as any).value || '')

      if (value && (encoding === 'safeStorage' || encoding === 'plain' || !encoding)) {
        out[headerName] = { encoding: encoding || 'plain', value }
      }
    }
  }

  return out
}

function remoteRequestMatchesBaseUrl(requestUrl, baseUrl) {
  try {
    const request = new URL(requestUrl)
    const base = new URL(baseUrl)
    const basePath = base.pathname.replace(/\/+$/, '')

    const requestProtocol =
      request.protocol === 'ws:' ? 'http:' : request.protocol === 'wss:' ? 'https:' : request.protocol

    const baseProtocol = base.protocol === 'ws:' ? 'http:' : base.protocol === 'wss:' ? 'https:' : base.protocol

    if (requestProtocol !== baseProtocol || request.host !== base.host) {
      return false
    }

    return !basePath || request.pathname === basePath || request.pathname.startsWith(`${basePath}/`)
  } catch {
    return false
  }
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

  // Tolerate a pasted command: "ssh root@box" → "root@box".
  host = host.replace(/^ssh\s+/i, '').trim()

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

  // A Desktop profile can be a local routing label rather than the profile
  // name used by the remote Hermes installation. Preserve an explicit mapping
  // when it is a valid Hermes profile identifier; otherwise fall back to the
  // historical same-name behavior in the caller.
  const remoteProfile = String(entry.remoteProfile || '').trim()

  if (/^[a-z0-9][a-z0-9_-]{0,63}$/.test(remoteProfile) && !RESERVED_REMOTE_PROFILES.has(remoteProfile)) {
    out.remoteProfile = remoteProfile
  }

  // Slow remotes can need longer than the default for a cold `hermes serve`
  // to announce its ready port (#97264). Accepted as a positive integer
  // number of milliseconds; a numeric string (e.g. "180000" from a hand-edited
  // config file) is coerced via Number() like the port field above. Values
  // above READY_TIMEOUT_MS_CAP are clamped to the cap rather than rejected —
  // the intent (wait longer) is real, the magnitude may be a typo — and
  // anything non-positive or non-numeric falls back to the default in
  // remote-lifecycle.ts.
  const readyTimeoutMs = Number(entry.readyTimeoutMs)

  if (Number.isInteger(readyTimeoutMs) && readyTimeoutMs > 0) {
    out.readyTimeoutMs = Math.min(readyTimeoutMs, READY_TIMEOUT_MS_CAP)
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
 * `url`. Pure: `token` and `headers` are raw stored secrets; main.ts decrypts
 * them. Returns `{ url, authMode, token, headers } | null`.
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

  const headers = normalizeRemoteHeaders(entry.headers)

  return {
    url,
    authMode: normAuthMode(entry.authMode),
    token: entry.token,
    ...(Object.keys(headers).length > 0 ? { headers } : {})
  }
}

export interface ProfileRouteOptions {
  /** Profile name on a separately-scoped backend when it differs from the
   * desktop's local routing label (managed SSH `remoteProfile`). */
  backendProfile?: null | string
  globalRemote?: boolean
  primaryProfile?: null | string
  profileRemoteOverride?: boolean
  /** The primary profile's own backend resolves to a remote host. */
  primaryRemoteActive?: boolean
  /** A stored per-profile entry exists for this profile (local or remote). */
  ownEntry?: boolean
  requestMethod?: null | string
  requestPath?: null | string
}

export interface ProfileBackendRoute {
  /** Which backend serves this profile: the window backend, or a pooled one. */
  backend: 'pool' | 'primary'
  /**
   * Profile to tag on the returned descriptor when the backend is shared and
   * therefore not itself scoped to that profile. Null when the backend already
   * belongs to the profile.
   */
  descriptorProfile: null | string
  /** Whether REST paths on this route must carry `?profile=` to be scoped. */
  scopePath: boolean
}

const LOCAL_PRIMARY_SCOPED_ROUTES = new Set([
  'GET /api/config',
  'PUT /api/config',
  'GET /api/config/raw',
  'PUT /api/config/raw',
  'GET /api/config/schema',
  'DELETE /api/env',
  'GET /api/env',
  'PUT /api/env',
  'POST /api/env/reveal',
  'GET /api/model/auxiliary',
  'GET /api/model/info',
  'GET /api/model/moa',
  'PUT /api/model/moa',
  'GET /api/model/options',
  'POST /api/model/set',
  'GET /api/skills',
  'GET /api/skills/content',
  'PUT /api/skills/toggle',
  'POST /api/skills/hub/install',
  'GET /api/skills/hub/official',
  'GET /api/skills/hub/preview',
  'GET /api/skills/hub/scan',
  'GET /api/skills/hub/search',
  'GET /api/skills/hub/sources',
  'POST /api/skills/hub/uninstall',
  'POST /api/skills/hub/update',
  // Spawns a background action polled via /api/actions/{name}/status — must
  // live on the SAME backend as that poll family (below), or the poll asks a
  // backend that never registered the dynamic action name and 404s.
  'POST /api/mcp/catalog/install'
])

function localPrimaryRequestScope(opts: ProfileRouteOptions): boolean | null {
  const rawPath = String(opts.requestPath || '')

  if (!rawPath) {
    return null
  }

  let pathname

  try {
    pathname = new URL(rawPath, 'https://example.invalid').pathname
  } catch {
    return null
  }

  const method = String(opts.requestMethod || 'GET').toUpperCase()

  if (LOCAL_PRIMARY_SCOPED_ROUTES.has(`${method} ${pathname}`)) {
    return true
  }

  // Action-status polls MUST land on the same backend as the endpoints that
  // spawned them: `_spawn_hermes_action` registers the (often dynamic, e.g.
  // `skills-install-<slug>-<hash>`) action name only in the spawning
  // process's memory. Every action-spawning route above scopes to the
  // primary, so the poll family follows — a pooled-backend poll 404s with
  // "Unknown action" even though the install itself succeeded (#89xxx).
  if (pathname.startsWith('/api/actions/')) {
    return true
  }

  // Every current /api/tools handler accepts `profile`; every /api/profiles
  // handler either aggregates profiles or names its target in the path/body.
  // These are the only whole families safe to route through the primary.
  if (pathname === '/api/tools' || pathname.startsWith('/api/tools/')) {
    return true
  }

  if (pathname === '/api/profiles' || pathname.startsWith('/api/profiles/')) {
    return false
  }

  return null
}

/**
 * The one place that answers "which backend serves profile P, and does its
 * REST path need a profile scope?". Six routes, in precedence order:
 *
 *  1. The primary profile owns a local/window backend outright; on a global
 *     remote its label is still carried per request because launch home can
 *     differ from the selected profile.
 *  2. A profile with its own remote override gets a pooled descriptor for that
 *     host, which is already scoped to it.
 *  3. A profile inheriting the app-global remote shares the primary backend —
 *     one host serves every profile — so it is scoped per request instead.
 *  4. An unknown profile under a remote primary shares that remote backend.
 *     A stored local profile remains isolated in its own backend.
 *  5. A local profile REST request that the primary backend can safely scope
 *     reuses that backend, with `?profile=` when the handler accepts it.
 *  6. Any other local profile gets its own pooled backend, spawned with
 *     `--profile`, so its `HERMES_HOME` scopes it.
 *
 * Routing used to be spread across three overlapping predicates that each
 * re-derived part of this table, which is how case 3 ended up registering
 * reapable pool entries for backends it never owned.
 */
function resolveProfileBackendRoute(profile, opts: ProfileRouteOptions = {}): ProfileBackendRoute {
  const scopedProfile = connectionScopeKey(profile)
  const primaryProfile = connectionScopeKey(opts.primaryProfile) || 'default'

  if (!scopedProfile) {
    return { backend: 'primary', descriptorProfile: null, scopePath: false }
  }

  if (scopedProfile === primaryProfile) {
    // A global remote is a multi-profile dashboard, not a backend process
    // launched for this Desktop label. Even its "primary" label must travel on
    // the wire: the dashboard's process HERMES_HOME can belong to a different
    // launch profile, so a bare request silently reads that profile instead.
    return opts.globalRemote
      ? { backend: 'primary', descriptorProfile: scopedProfile, scopePath: true }
      : { backend: 'primary', descriptorProfile: null, scopePath: false }
  }

  if (opts.profileRemoteOverride) {
    return { backend: 'pool', descriptorProfile: null, scopePath: false }
  }

  if (opts.globalRemote) {
    return { backend: 'primary', descriptorProfile: scopedProfile, scopePath: true }
  }

  if (opts.primaryRemoteActive) {
    if (!opts.ownEntry) {
      // The primary profile's own backend is a remote gateway (per-profile
      // override or env) and this sub-profile has no stored entry of its own.
      // Route through that gateway with profile scoping instead of spawning a
      // fresh local backend that shares nothing but the name (#88296).
      return { backend: 'primary', descriptorProfile: scopedProfile, scopePath: true }
    }

    // A stored local profile must not be redirected into the remote primary,
    // even when its REST endpoint supports profile scoping.
    return { backend: 'pool', descriptorProfile: null, scopePath: false }
  }

  const localScope = localPrimaryRequestScope(opts)

  if (localScope !== null) {
    return {
      backend: 'primary',
      descriptorProfile: localScope ? scopedProfile : null,
      scopePath: localScope
    }
  }

  return { backend: 'pool', descriptorProfile: null, scopePath: false }
}

/**
 * Reconcile the renderer's desktop-facing profile label with the backend's
 * profile namespace, then add `request.profile` when a shared backend needs it.
 *
 * A managed SSH override can deliberately map local `mara` to remote `default`.
 * Endpoint-level filters (cron list / blueprint instantiate) arrive as an
 * explicit `?profile=mara`; translate only that self-scope. Cross-profile
 * selectors such as `all` or another concrete profile retain their meaning.
 */
function pathWithGlobalRemoteProfile(path, profile, opts: ProfileRouteOptions = {}) {
  const translated = translateSelfProfileQuery(path, profile, opts.backendProfile)

  if (translated !== path) {
    return translated
  }

  if (!resolveProfileBackendRoute(profile, opts).scopePath) {
    return path
  }

  return pathWithProfileScope(path, profile)
}

/** Extra profile-valued query keys, beyond `profile`, that name the same
 *  self-scope on a given path. The sidebar batches recents/cron/messaging
 *  behind `recents_profile` instead of `profile`, so an SSH alias rewrite
 *  that only looks at `?profile=` leaves those reads on the remote default. */
const SELF_PROFILE_QUERY_KEYS_BY_PATH: Record<string, string[]> = {
  '/api/profiles/sessions/sidebar': ['recents_profile']
}

/**
 * Translate an explicit self-profile query from a Desktop routing alias to the
 * backend's own profile namespace (a managed SSH `remoteProfile` can map local
 * `mara` to remote `default`). Only endpoint-declared profile-valued params
 * equal to the alias itself are rewritten; cross-profile selectors (`all`,
 * another concrete profile) and unfiltered paths pass through untouched. Used
 * by the v1 profile route above and by the registry SSH branch of the
 * `hermes:api` handler — both routes reach a backend whose namespace is the
 * remote profile, not the alias.
 */
function translateSelfProfileQuery(path, profile, backendProfile) {
  const scopedProfile = connectionScopeKey(profile)
  const backend = connectionScopeKey(backendProfile)

  if (!scopedProfile || !backend || backend === scopedProfile) {
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

  const profileQueryKeys = ['profile', ...(SELF_PROFILE_QUERY_KEYS_BY_PATH[parsed.pathname] || [])]
  let changed = false

  for (const key of profileQueryKeys) {
    if (connectionScopeKey(parsed.searchParams.get(key)) !== scopedProfile) {
      continue
    }

    parsed.searchParams.set(key, backend)
    changed = true
  }

  if (!changed) {
    return path
  }

  return `${parsed.pathname}${parsed.search}${parsed.hash}`
}

/**
 * Unconditionally scope a REST path to a profile via `?profile=`. Used by the
 * global-remote route above and by registry `sharedRemote` connections (one
 * gateway host serving every profile, scoped per request). An explicit
 * `?profile=` already on the path wins; an empty profile is a no-op.
 */
function pathWithProfileScope(path, profile) {
  const scopedProfile = connectionScopeKey(profile)

  if (!scopedProfile) {
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

export interface RegistryBackendRequestScope {
  remoteProfile?: null | string
  sharedRemote?: boolean
}

/**
 * Scope a REST path for a resolved registry backend. Shared remotes serve
 * multiple profiles from one process and need an explicit profile query;
 * isolated SSH backends already own one profile but may translate a Desktop
 * alias in an existing self-profile filter.
 */
function pathForRegistryBackendRequest(path, profile, backend: RegistryBackendRequestScope) {
  return backend.sharedRemote
    ? pathWithProfileScope(path, profile)
    : translateSelfProfileQuery(path, profile, backend.remoteProfile)
}

/**
 * Registry connection a REST request is explicitly pinned to, or null for the
 * legacy profile-routed path. An explicit `local` id must stay registry-scoped:
 * when the v1 route is remote, only the registry resolver can force the request
 * back to this device. Single-source users omit the id and keep the
 * byte-identical v1 route.
 */
function apiRequestRegistryConnectionId(request): null | string {
  const raw = request && typeof request === 'object' ? (request as { connectionId?: unknown }).connectionId : ''
  const id = String(raw ?? '').trim()

  if (!id) {
    return null
  }

  return id
}

export interface ProfileApiRequestRoute {
  /** Profile passed to ensureBackend; null selects the primary backend. */
  backendProfile: null | string
  requestPath: string
}

/**
 * Resolve the two decisions made by the `hermes:api` IPC handler from the same
 * routing table: which backend serves the request, and whether its URL needs a
 * profile query scope.
 */
function resolveProfileApiRequest(profile, path, opts: ProfileRouteOptions = {}): ProfileApiRequestRoute {
  const scopedProfile = connectionScopeKey(profile)
  const requestPath = String(path || '')
  const routeOpts = { ...opts, requestPath }
  const route = resolveProfileBackendRoute(scopedProfile, routeOpts)

  return {
    backendProfile: route.backend === 'pool' ? scopedProfile : null,
    requestPath: pathWithGlobalRemoteProfile(requestPath, scopedProfile, routeOpts)
  }
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

/**
 * True only when the short-lived Privy ACCESS token (`privy-token`) is present
 * — the exact cookie `/api/agents` validates. A jar can satisfy
 * `cookiesHavePrivySession` (renewable session: `privy-session` /
 * `privy-refresh-token`) while failing this check; that gap is the cold-start
 * "Signed in" + "No agents found" contradiction, and the signal that a silent
 * renewal (not an interactive re-login) is the right recovery (#73495).
 */
function cookiesHavePrivyAccessToken(cookies) {
  if (!Array.isArray(cookies)) {
    return false
  }

  return cookies.some(c => c && c.value && PRIVY_ACCESS_COOKIE_VARIANTS.includes(c.name))
}

export {
  apiRequestRegistryConnectionId,
  AT_COOKIE_VARIANTS,
  authModeFromStatus,
  buildGatewayWsUrl,
  buildGatewayWsUrlWithTicket,
  connectionScopeKey,
  cookiesHaveLiveSession,
  cookiesHavePrivyAccessToken,
  cookiesHavePrivySession,
  cookiesHaveSession,
  gatewayTicketFailure,
  gatewayWsUrlIpcResult,
  hostLabelFromBaseUrl,
  isGatewayAuthRejection,
  localProfileEntry,
  modeIsRemoteLike,
  normalizeRemoteBaseUrl,
  normalizeRemoteHeaders,
  normalizeSshConfig,
  normAuthMode,
  pathForRegistryBackendRequest,
  pathWithGlobalRemoteProfile,
  pathWithProfileScope,
  PRIVY_ACCESS_COOKIE_VARIANTS,
  PRIVY_SESSION_COOKIE_VARIANTS,
  profileHasRemoteConnection,
  profileRemoteOverride,
  profileSshOverride,
  remoteRequestMatchesBaseUrl,
  resolveAuthMode,
  resolveProfileApiRequest,
  resolveProfileBackendRoute,
  resolveRemoteSshDashboardProfile,
  resolveTestWsUrl,
  RT_COOKIE_VARIANTS,
  savedProfileSsh,
  tokenPreview,
  translateSelfProfileQuery,
  withTransientRetries
}
