import fs from 'node:fs'
import path from 'node:path'

import { readJsonErrorBody, readStatusCode } from './api-transport'
import { discoverWithTeamFallback } from './cloud-discovery'
import { buildGatewayWsUrlWithTicket, isGatewayAuthRejection, normalizeRemoteBaseUrl, withTransientRetries } from './connection-config'
import { createNativeAccessTokenCoordinator, NativeAuthChangedError } from './native-access-token'
import { resolveJsonBody } from './native-auth-decisions'
import { nativeRefreshUrl, type NativeTokenSet, parseTokenResponse, tokenNeedsRefresh } from './native-oauth'
import { loadNativeTokenSet, type NativeTokenStoreIo, persistNativeTokenSet } from './native-token-store'
import { mintGatewayWsTicket as mintOauthGatewayWsTicket } from './oauth-rest-request'
import { createPortalSession } from './portal-session'

export function createDesktopConnectionAuthRuntime(deps: {
  app: any
  BrowserWindow: any
  fetchJson: (url: string, token: any, options?: any) => Promise<any>
  fetchJsonViaOauthSession: (url: string, options?: any) => Promise<any>
  encryptDesktopSecret: (value: any, options?: any) => any
  decryptDesktopSecret: (secret: any) => string
  ensureBackend: (profile?: any) => Promise<any>
  getOauthSession: () => any
  warmOauthCookieStore: () => Promise<unknown>
  hasOauthSessionCookie: (url: string) => Promise<boolean>
  openOauthLoginWindow: (url: string, options?: any) => Promise<any>
  rememberRemoteWsHeaders: (url: string, headers?: any) => void
  rememberLog: (message: string) => void
}) {
  const { app, BrowserWindow, fetchJson, fetchJsonViaOauthSession, encryptDesktopSecret, decryptDesktopSecret, ensureBackend, getOauthSession, warmOauthCookieStore, hasOauthSessionCookie, openOauthLoginWindow, rememberRemoteWsHeaders, rememberLog } = deps
// ---------------------------------------------------------------------------
// RFC 8252 native-app tokens (system-browser + loopback + PKCE).
//
// Unlike the cookie flow, the native flow hands the desktop opaque bearer
// tokens it holds itself: the access token authenticates REST via
// ``Authorization: Bearer`` (which the gateway gate now accepts) and mints WS
// tickets the same way, so NO browser session cookie or embedded webview is
// involved. Tokens are persisted encrypted at rest via Electron ``safeStorage``
// (OS keychain) keyed by gateway base URL, and refreshed via
// ``/auth/native/refresh`` before expiry. This is the desktop half of the
// feature; the server half lives in hermes_cli/dashboard_auth/native_flow.py.
// ---------------------------------------------------------------------------

// In-memory cache of decrypted native tokens, keyed by normalized base URL.
// Backed by the encrypted on-disk store so it survives restarts.
const _nativeTokens = new Map<string, NativeTokenSet>()

function _nativeTokenStorePath() {
  // Co-located with the connection config under userData; one JSON file mapping
  // baseUrl → { encoding, value } safeStorage payloads.
  return path.join(app.getPath('userData'), 'native-oauth-tokens.json')
}

// The electron-coupled half of the token store: safeStorage encryption plus the
// userData file. native-token-store.ts owns the serialization/parse round trip
// so it can be tested without an Electron runtime.
function _nativeTokenStoreIo(): NativeTokenStoreIo {
  return {
    encrypt: encryptDesktopSecret,
    decrypt: decryptDesktopSecret,
    readStoreText: () => fs.readFileSync(_nativeTokenStorePath(), 'utf8'),
    writeStoreText: (text: string) => {
      fs.mkdirSync(path.dirname(_nativeTokenStorePath()), { recursive: true })
      fs.writeFileSync(_nativeTokenStorePath(), text, { mode: 0o600 })
    },
    rememberLog
  }
}

function _persistNativeTokens(baseUrl: string, tokens: NativeTokenSet | null) {
  persistNativeTokenSet(baseUrl, tokens, _nativeTokenStoreIo())
}

function _loadNativeTokens(baseUrl: string): NativeTokenSet | null {
  baseUrl = normalizeRemoteBaseUrl(baseUrl)
  const cached = _nativeTokens.get(baseUrl)

  if (cached) {
    return cached
  }

  const tokens = loadNativeTokenSet(baseUrl, _nativeTokenStoreIo())

  if (tokens) {
    _nativeTokens.set(baseUrl, tokens)
  }

  return tokens
}

function _storeNativeTokens(baseUrl: string, tokens: NativeTokenSet) {
  _persistNativeTokens(baseUrl, tokens)
  _nativeTokens.set(baseUrl, tokens)
}

function _clearNativeTokens(baseUrl: string) {
  _nativeTokens.delete(baseUrl)
  _persistNativeTokens(baseUrl, null)
}

// True when we hold native bearer tokens for this gateway (the native-flow
// analogue of hasLiveOauthSession's cookie check).
function hasNativeSession(baseUrl: string): boolean {
  return _loadNativeTokens(baseUrl) !== null
}

// POST JSON WITHOUT the OAuth cookie partition — used for the native token +
// refresh exchanges, which are cookieless by design. Thin wrapper over
// fetchJson (no token) so it shares timeout/JSON handling.
function postJsonNoAuth(url: string, body: unknown, opts: any = {}) {
  // resolveJsonBody passes the object through UNCHANGED — fetchJson owns
  // JSON.stringify. Pre-stringifying here double-encodes the body (a JSON
  // string inside a JSON string), which the gateway's Pydantic model rejects
  // with a 422 "Input should be a valid dictionary" (the native
  // /auth/native/token + /auth/native/refresh legs both go through here).
  return fetchJson(url, null, { method: 'POST', body: resolveJsonBody(body), ...opts })
}

// All explicit mutations go through the coordinator; only its refresh/store
// dependencies may call the raw persistence helpers above.
const nativeAccessTokenCoordinator = createNativeAccessTokenCoordinator({
  clearTokens: _clearNativeTokens,
  isRefreshAuthRejection: error => readStatusCode(error) === 401,
  loadTokens: _loadNativeTokens,
  normalizeBaseUrl: normalizeRemoteBaseUrl,
  refreshTokens: async (baseUrl, tokens) =>
    parseTokenResponse(
      await postJsonNoAuth(
        nativeRefreshUrl(baseUrl),
        { refresh_token: tokens.refreshToken, provider: tokens.provider },
        { timeoutMs: 10_000 }
      )
    ),
  storeTokens: _storeNativeTokens,
  tokenNeedsRefresh
})

const ensureNativeAccessToken = nativeAccessTokenCoordinator.ensure

// Mint a single-use WS ticket for a gated gateway.
// Ticket POSTs are replay-safe; arbitrary REST mutations never use this retry loop.
async function mintGatewayWsTicket(baseUrl, headers = {}) {
  return withTransientRetries(
    () =>
      mintOauthGatewayWsTicket(
        baseUrl,
        {
          ensureNativeAccessToken,
          fetchJson,
          fetchJsonViaOauthSession
        },
        headers
      ),
    {
      isRetryable: (error: unknown) => !(error instanceof NativeAuthChangedError) && !isGatewayAuthRejection(error)
    }
  )
}

// Build a fresh WS URL for the *current* connection. Critical for reconnects:
// OAuth WS tickets are single-use with a ~30s TTL, so the ticket baked into
// the cached connection's wsUrl is stale on the second connect. The renderer
// calls this immediately before every gateway.connect() so each WS upgrade
// carries a freshly-minted ticket. For local/token connections this just
// reuses the static token (no minting needed).
async function freshGatewayWsUrl(profile) {
  // Mint for the requested profile's backend, NOT always the primary. The
  // renderer re-mints right before every gateway.connect(); when swapping to a
  // pooled profile we must return THAT backend's ws URL, otherwise the connect
  // silently lands back on the primary (default) backend and writes sessions to
  // the wrong profile's DB. A null/empty profile resolves to the primary, so
  // legacy callers and single-profile users are unchanged.
  const connection = await ensureBackend(profile)

  if (connection.authMode === 'oauth') {
    const ticket = await mintGatewayWsTicket(connection.baseUrl, connection.headers)
    const wsUrl = buildGatewayWsUrlWithTicket(connection.baseUrl, ticket)

    rememberRemoteWsHeaders(wsUrl, connection.headers)

    return wsUrl
  }

  // Local/token: the cached wsUrl already carries the (long-lived) token.
  rememberRemoteWsHeaders(connection.wsUrl, connection.headers)

  return connection.wsUrl
}

// --- Hermes Cloud discovery + silent per-agent sign-in (cloud-auto-discovery
// Phase 3) ---------------------------------------------------------------
//
// The "cloud" connection mode lets a user sign in to the Nous portal ONCE in
// the OAuth session partition, then (a) discover their hosted agents and (b)
// connect to any of them with no second interactive sign-in. Both ride the one
// portal session cookie living in `persist:hermes-remote-oauth`:
//   - discovery  → GET {portal}/api/agents over the partition-bound net; the
//     portal session cookie authenticates it (NAS Phase 2.5 accepts the cookie).
//   - cascade    → opening an agent's own /login in the same partition hits the
//     portal's silent auto-approve (org member, existing session) and 302s back
//     with that agent's session cookie — no prompt. Each agent still completes
//     its own PKCE exchange; SSO removes the human click, not a security check.

// Canonical Nous portal base URL, overridable for staging/dev. Mirrors the CLI
// convention (hermes_cli/auth.py DEFAULT_NOUS_PORTAL_URL + the same env names)
// so a single override flips every Hermes surface to the same portal.
const DEFAULT_NOUS_PORTAL_URL = 'https://portal.nousresearch.com'

function resolvePortalBaseUrl() {
  const raw = process.env.HERMES_PORTAL_BASE_URL || process.env.NOUS_PORTAL_BASE_URL || DEFAULT_NOUS_PORTAL_URL

  return String(raw).trim().replace(/\/+$/, '')
}

const { hasLivePortalSession, hasPortalAccessToken, renewPortalAccessSilently, openPortalLoginWindow } =
  createPortalSession({
    isReady: () => app.isReady(),
    getOauthSession,
    resolvePortalBaseUrl,
    warmOauthCookieStore,
    createWindow: options => new BrowserWindow(options),
    rememberLog
  })

// Discover the hosted (Hermes Cloud) agents the signed-in user can see. Calls
// the NAS trimmed-summary endpoint over the partition-bound net, so the portal
// session cookie is attached automatically (no bearer needed — NAS accepts the
// cookie). Returns { agents } on success, or { needsOrgSelection: true, orgs }
// when the user belongs to multiple orgs and hasn't picked one yet (NAS 409
// org_selection_required). Pass `org` (a slug/id from a prior org list) to
// scope discovery to that org. Throws a needsCloudLogin-tagged error when no
// portal session is present.
async function discoverCloudAgents(org?: string) {
  const portalBaseUrl = resolvePortalBaseUrl()

  if (!(await hasLivePortalSession())) {
    const err = new Error(
      'You are not signed in to Hermes Cloud. Open Settings → Gateway, choose Hermes Cloud, and sign in.'
    ) as any

    err.needsCloudLogin = true
    throw err
  }

  // Access cookies expire before refresh credentials. Let the portal renew
  // whichever session this browser currently holds before discovery.
  if (!(await hasPortalAccessToken())) {
    await renewPortalAccessSilently()
  }

  let body

  const fetchAgents = () =>
    discoverWithTeamFallback(
      selectedOrg =>
        fetchJsonViaOauthSession(
          `${portalBaseUrl}/api/agents${selectedOrg ? `?org=${encodeURIComponent(selectedOrg)}` : ''}`,
          {
            method: 'GET',
            timeoutMs: 15_000
          }
        ),
      org
    )

  try {
    body = (await fetchAgents()) as any
  } catch (initialError) {
    let error = initialError as any

    // A 401 with renewal material still in the jar: attempt ONE bounded silent
    // renewal and retry, so a lapsed access token doesn't surface as a full
    // interactive re-login while a 30-day refresh session sits unused. Only a
    // rejected/failed renewal (or a second 401 on genuinely fresh access)
    // falls through to needsCloudLogin.
    if (error && error.statusCode === 401 && (await renewPortalAccessSilently({ force: true }))) {
      try {
        body = (await fetchAgents()) as any
      } catch (retryError) {
        error = retryError
      }
    }

    if (body === undefined) {
      // A 401 means the portal session lapsed (and silent renewal could not
      // recover it) — surface it as a re-login, not a generic failure.
      if (error && error.statusCode === 401) {
        const err = new Error(
          'Your Hermes Cloud session has expired. Open Settings → Gateway and sign in again.'
        ) as any

        err.needsCloudLogin = true
        err.cause = error
        throw err
      }

      // A 409 means we're a multi-org user who hasn't picked an org. The body
      // carries the user's org list; surface it so the renderer shows a picker
      // and re-calls discovery with the chosen org. (fetchJsonViaOauthSession
      // throws on >=400 with err.statusCode + err.message "409: <json body>".)
      if (error && error.statusCode === 409) {
        const orgs = parseOrgSelectionError(error)

        if (orgs) {
          return { needsOrgSelection: true, orgs }
        }
      }

      throw error
    }
  }

  return { agents: trimCloudAgents(body), org: trimCloudOrg(body?.org) }
}

// Project a NAS response org ({ id, slug, name, isPersonal }) to the trimmed
// shape the renderer persists, or null when absent/malformed.
function trimCloudOrg(org) {
  if (!org || typeof org !== 'object' || typeof org.id !== 'string') {
    return null
  }

  return {
    id: org.id,
    slug: typeof org.slug === 'string' ? org.slug : null,
    name: typeof org.name === 'string' ? org.name : org.id,
    isPersonal: Boolean(org.isPersonal),
    role: typeof org.role === 'string' ? org.role : 'MEMBER'
  }
}

// Extract the org list from a 409 org_selection_required error body. Parse
// defensively and return null if it isn't the shape we expect (caller then
// rethrows).
function parseOrgSelectionError(error) {
  const parsed = readJsonErrorBody(error)

  if (parsed?.error !== 'org_selection_required' || !Array.isArray(parsed.orgs)) {
    return null
  }

  return parsed.orgs
    .filter(o => o && typeof o === 'object' && typeof o.id === 'string')
    .map(o => ({
      id: o.id,
      slug: typeof o.slug === 'string' ? o.slug : null,
      name: typeof o.name === 'string' ? o.name : o.id,
      isPersonal: Boolean(o.isPersonal),
      role: typeof o.role === 'string' ? o.role : 'MEMBER'
    }))
}

// Project NAS's agent rows to the trimmed DTO the renderer consumes.
function trimCloudAgents(body) {
  const agents = Array.isArray(body?.agents) ? body.agents : []

  return agents
    .filter(a => a && typeof a === 'object' && typeof a.id === 'string')
    .map(a => ({
      id: a.id,
      name: typeof a.name === 'string' ? a.name : a.id,
      status: typeof a.status === 'string' ? a.status : 'unknown',
      dashboardUrl: typeof a.dashboardUrl === 'string' ? a.dashboardUrl : null,
      dashboardGatewayState: typeof a.dashboardGatewayState === 'string' ? a.dashboardGatewayState : 'unknown'
    }))
}

// Silent per-agent sign-in: open the selected agent dashboard's /login in the
// SAME OAuth partition. Because the user already holds a live portal session
// there, the agent's /oauth/authorize auto-approves (org member) and 302s back,
// setting that agent's gateway session cookie WITHOUT a second interactive
// prompt. Reuses openOauthLoginWindow — the window self-closes the instant the
// agent's session cookie lands (a silent flow finishes in well under a second;
// if the portal session were absent it would fall through to an interactive
// login, which the discovery gate already prevents). Returns once the agent's
// gateway session cookie is present.
async function cloudAgentSilentSignIn(dashboardUrl) {
  const baseUrl = normalizeRemoteBaseUrl(dashboardUrl)

  // Pre-req: a live portal session must exist, or this would surface an
  // interactive prompt rather than a silent cascade. Discovery already gates on
  // this, but a selection can arrive after the session lapsed.
  if (!(await hasLivePortalSession())) {
    const err = new Error('Your Hermes Cloud session has expired. Sign in to Hermes Cloud again.') as any
    err.needsCloudLogin = true
    throw err
  }

  // The cascade rides the portal's auto-approve, which needs the short-lived
  // access state just like discovery. If only renewal material survived the
  // restart, mint a fresh access token first so the hidden cascade window
  // auto-SSOs instead of stalling on an interactive chooser (#73495).
  if (!(await hasPortalAccessToken())) {
    await renewPortalAccessSilently()
  }

  await openOauthLoginWindow(baseUrl, { silent: true })

  return { baseUrl, connected: await hasOauthSessionCookie(baseUrl) }
}


  return {
    _nativeTokenStoreIo,
    nativeAccessTokenCoordinator,
    ensureNativeAccessToken,
    hasNativeSession,
    postJsonNoAuth,
    mintGatewayWsTicket,
    freshGatewayWsUrl,
    resolvePortalBaseUrl,
    hasLivePortalSession,
    hasPortalAccessToken,
    renewPortalAccessSilently,
    openPortalLoginWindow,
    discoverCloudAgents,
    cloudAgentSilentSignIn
  }
}
