/**
 * apex-managed.cjs
 *
 * Pure, electron-free helpers for the ApexNodes "managed LLM" default path
 * (Desktop V0.2). Kept standalone (no `require('electron')`) so it can be
 * unit-tested with `node --test`, same pattern as connection-config.cjs /
 * dashboard-token.cjs. main.cjs requires these and wires them into the
 * electron-coupled IPC + boot layer.
 *
 * Background — the two independent "login/connection" systems this app has:
 *
 *   1. REMOTE GATEWAY (connection-config.cjs / dashboard-token.cjs /
 *      oauth-net-request.cjs / gateway-ws-probe.cjs): connects the desktop to a
 *      *remote orchestration backend* (the dashboard/gateway that runs agent
 *      sessions) via HttpOnly session cookies + /api/ws tickets. This is a
 *      power-user feature, OFF by default; the desktop runs a LOCAL runtime.
 *
 *   2. MANAGED LLM (this module): points the LOCAL runtime's inference at the
 *      ApexNodes relay so a signed-in user gets zero-key chat. The relay is an
 *      OpenAI-compatible endpoint that bills the user's cloud account; the
 *      runtime just needs `model.base_url` + `model.api_key` + `model.default`
 *      in config.yaml (the exact same three fields the "Local / custom
 *      endpoint" BYOK flow already writes via /api/model/set).
 *
 * These are different concerns with different credentials — this module is ONLY
 * about (2). It never touches the remote-gateway cookie machinery.
 *
 * ── Backend contract (P0, confirmed) ────────────────────────────────────────
 * The relay validates an *agent-scoped* key from the cloud `api_keys` table; the
 * raw key is minted server-side and is never returned by the generic APIs (login
 * yields a JWT; the dashboard "API Keys" page mints a `user_api_keys` capability
 * key the relay does NOT accept). So a dedicated provisioning endpoint mints a
 * relay-VALID key for a logged-in user:
 *
 *   POST {API_BASE}/api/v1/desktop/provision-key
 *   Authorization: Bearer <login JWT>
 *   body: {}
 *   200 → { api_key, base_url, model }
 *
 * The desktop uses base_url + model FROM THE RESPONSE (never hardcoded) so the
 * server stays the source of truth for routing; the local DEFAULT_* constants
 * are only a fallback for display/seed when no response is on hand. If the
 * endpoint isn't reachable (not deployed, network error),
 * `resolveManagedRelayCredential` returns '' and the desktop transparently
 * falls back to the BYOK onboarding (no regression).
 */

// ── ApexNodes default endpoints ─────────────────────────────────────────────
// All overridable via env so a staging build can retarget without a code change
// (mirrors how main.cjs lets HERMES_DESKTOP_* env vars override prod defaults).

// User-facing site host. Login lives under `${AUTH_BASE}/api/v1/auth/...`. The
// relay public path also lives on this host (nginx `location /relay/` →
// 127.0.0.1:7000, prefix stripped).
const DEFAULT_AUTH_BASE = 'https://apex-nodes.com'

// API host for machine endpoints, incl. the desktop key-provisioning route
// (`POST {API_BASE}/api/v1/desktop/provision-key`). Both hosts route /api/ to
// the same backend, but the contract pins provision-key on the api. host.
const DEFAULT_API_BASE = 'https://api.apex-nodes.com'

// Fallback relay public base_url for `model.base_url` — used only for the
// display/seed when no provision-key response is on hand; the live path uses the
// base_url returned by provision-key. The relay's OpenAI-compatible chat route
// is `/v1/chat/completions`; nginx strips the `/relay` prefix, so the public
// form that reaches it is `https://apex-nodes.com/relay/v1/chat/completions`.
// The runtime appends `/chat/completions`, hence the base ends at `/relay/v1`.
const DEFAULT_RELAY_BASE_URL = 'https://apex-nodes.com/relay/v1'

// Real model the relay routes to (our master key). hc-184 decouples the routed
// model from the `model` field the runtime sends — the relay routes by DB
// truth — but we still send the real id for clarity/correctness.
const DEFAULT_MANAGED_MODEL = 'deepseek-v4-pro'

// Display-only label shown in the UI. The underlying route is unchanged; this
// is a cosmetic mapping so users see the ApexNodes-branded model name.
const MANAGED_MODEL_DISPLAY = 'deepseek-v4-pro-APEX'

// The runtime treats the relay as a generic OpenAI-compatible endpoint, so the
// provider slug is the same `custom` the local/custom BYOK flow uses. Reusing
// `custom` means zero new runtime provider plumbing.
const MANAGED_PROVIDER = 'custom'

// Endpoint paths. LOGIN_PATH / REGISTER_PATH are on AUTH_BASE; PROVISION_KEY_PATH
// is on API_BASE. GOOGLE_START_PATH is the backend's browser OAuth entry (on
// API_BASE — see the shared login-rework contract).
const LOGIN_PATH = '/api/v1/auth/login'
const REGISTER_PATH = '/api/v1/auth/register'
const PROVISION_KEY_PATH = '/api/v1/desktop/provision-key'
const GOOGLE_START_PATH = '/api/v1/auth/google/start'

// User-facing site path of the web login page. The desktop "用 APEX 登录" flow
// opens `${AUTH_BASE}/zh/login?desktop_cb=<loopback>&state=<s>`; the web login
// page honors desktop_cb and redirects the browser back to the loopback with the
// minted token. Locale-pinned to zh (the desktop is China-first).
const WEB_LOGIN_PATH = '/zh/login'

function trimTrailingSlash(value) {
  return String(value || '').replace(/\/+$/, '')
}

/**
 * Resolve the effective ApexNodes endpoints, applying env overrides. Pure: the
 * caller passes `process.env` (or a stub in tests) so this stays electron-free
 * and deterministic.
 *
 * @param {Record<string, string | undefined>} [env]
 * @returns {{ authBase: string, apiBase: string, relayBaseUrl: string,
 *             model: string, modelDisplay: string, provider: string,
 *             loginUrl: string, provisionKeyUrl: string }}
 */
function resolveApexEndpoints(env = {}) {
  const authBase = trimTrailingSlash(env.APEXNODES_AUTH_BASE || DEFAULT_AUTH_BASE)
  const apiBase = trimTrailingSlash(env.APEXNODES_API_BASE || DEFAULT_API_BASE)
  const relayBaseUrl = trimTrailingSlash(env.APEXNODES_RELAY_BASE_URL || DEFAULT_RELAY_BASE_URL)
  const model = String(env.APEXNODES_MANAGED_MODEL || DEFAULT_MANAGED_MODEL).trim() || DEFAULT_MANAGED_MODEL

  return {
    authBase,
    apiBase,
    relayBaseUrl,
    model,
    modelDisplay: MANAGED_MODEL_DISPLAY,
    provider: MANAGED_PROVIDER,
    loginUrl: `${authBase}${LOGIN_PATH}`,
    registerUrl: `${authBase}${REGISTER_PATH}`,
    provisionKeyUrl: `${apiBase}${PROVISION_KEY_PATH}`
  }
}

/**
 * Build the browser start URL for "用 Google 登录" (Deliverable 2). The desktop
 * opens this in the system browser; the backend bounces through Google and
 * redirects to the loopback `redirect_uri` with `?token=<JWT>&state=<state>`.
 * Lives on API_BASE per the shared contract.
 *
 * @param {string} redirectUri  the loopback callback (http://127.0.0.1:<port>/cb)
 * @param {string} state        random CSRF token echoed back on the callback
 * @param {Record<string, string | undefined>} [env]
 * @returns {string}
 */
function googleStartUrl(redirectUri, state, env = {}) {
  const { apiBase } = resolveApexEndpoints(env)
  const u = new URL(`${apiBase}${GOOGLE_START_PATH}`)
  u.searchParams.set('redirect_uri', String(redirectUri || ''))
  u.searchParams.set('state', String(state || ''))
  return u.toString()
}

/**
 * Build the browser start URL for "用 APEX 登录" (Deliverable 3). Opens the web
 * login page with `desktop_cb` + `state`; the web page redirects the browser
 * back to the loopback with `?token=<access_token>&state=<state>` after a
 * successful login/register. Lives on AUTH_BASE (the user-facing site).
 *
 * @param {string} redirectUri  the loopback callback (http://127.0.0.1:<port>/cb)
 * @param {string} state        random CSRF token echoed back on the callback
 * @param {Record<string, string | undefined>} [env]
 * @returns {string}
 */
function apexWebLoginUrl(redirectUri, state, env = {}) {
  const { authBase } = resolveApexEndpoints(env)
  const u = new URL(`${authBase}${WEB_LOGIN_PATH}`)
  u.searchParams.set('desktop_cb', String(redirectUri || ''))
  u.searchParams.set('state', String(state || ''))
  return u.toString()
}

/**
 * Parse + validate a browser loopback callback request URL for the Google / APEX
 * flows. The browser is redirected to `http://127.0.0.1:<port>/cb?token=<JWT>&state=<s>`.
 * We require the path to be `/cb`, the `state` to match the one we generated
 * (CSRF defense), and a non-empty `token`. Anything else → { ok:false, ... } so
 * the loopback handler can respond with an error page and never apply a token.
 *
 * Pure: the caller passes the request URL (Node sets req.url to a path+query for
 * an http server, so we parse against a dummy origin) and the expected state.
 *
 * @param {string} requestUrl   req.url from the loopback server (path + query)
 * @param {string} expectedState the state we generated when starting the flow
 * @returns {{ ok: true, token: string } | { ok: false, reason: string, isCallback: boolean }}
 */
function parseLoopbackCallback(requestUrl, expectedState) {
  let parsed
  try {
    // req.url is path-relative; a dummy origin lets URL parse the query.
    parsed = new URL(String(requestUrl || ''), 'http://127.0.0.1')
  } catch {
    return { ok: false, reason: 'invalid_request', isCallback: false }
  }
  // Only the /cb path is the OAuth callback. Other paths (e.g. /favicon.ico the
  // browser auto-requests) must be ignored, not treated as a failed login.
  const isCallback = parsed.pathname === '/cb'
  if (!isCallback) {
    return { ok: false, reason: 'not_callback', isCallback: false }
  }
  const error = parsed.searchParams.get('error')
  if (error) {
    return { ok: false, reason: error, isCallback: true }
  }
  const state = parsed.searchParams.get('state') || ''
  const expected = String(expectedState || '')
  // Constant-ish comparison; states are random opaque tokens, lengths usually
  // equal, so a plain !== is acceptable here (no secret-dependent branch leak of
  // value, only of equality — same as the rest of the OAuth state checks).
  if (!expected || state !== expected) {
    return { ok: false, reason: 'state_mismatch', isCallback: true }
  }
  const token = (parsed.searchParams.get('token') || '').trim()
  if (!token) {
    return { ok: false, reason: 'missing_token', isCallback: true }
  }
  return { ok: true, token }
}

/**
 * True when a redirect/callback URL targets the loopback interface (127.0.0.1 /
 * ::1 / localhost) over http. The desktop only ever points the browser at its
 * own loopback; this guards against accidentally opening a non-loopback start
 * URL (defense in depth — the backend MUST also validate redirect_uri).
 *
 * @param {string} url
 * @returns {boolean}
 */
function isLoopbackUrl(url) {
  let parsed
  try {
    parsed = new URL(String(url || ''))
  } catch {
    return false
  }
  if (parsed.protocol !== 'http:') return false
  // URL normalizes an IPv6 host to its bracketed form ("[::1]"); strip the
  // brackets so the bare-address comparison matches.
  const host = parsed.hostname.toLowerCase().replace(/^\[|\]$/g, '')
  return host === '127.0.0.1' || host === '::1' || host === 'localhost'
}

/**
 * True when the managed-LLM default path is enabled for this install. Single
 * gate the rest of the app (boot seed, onboarding, IPC) consults so the flag
 * check isn't scattered.
 *
 * Shipped ON by default now that the backend provision-key endpoint is live on
 * prod (2026-06-27): managed is the zero-key default. Every path still
 * auto-degrades to BYOK if provision-key is ever unreachable, so this never
 * breaks chat. Set `APEXNODES_MANAGED=0` (or false/no/off) to force the legacy
 * BYOK-first build.
 *
 * @param {Record<string, string | undefined>} [env]
 */
function isManagedEnabled(env = {}) {
  const raw = String(env.APEXNODES_MANAGED ?? '').trim().toLowerCase()
  return raw !== '0' && raw !== 'false' && raw !== 'no' && raw !== 'off'
}

/**
 * Build the config.yaml `model:` block for the managed relay path. The runtime
 * resolver reads `model.base_url` / `model.api_key` directly from config and
 * honors them for a `custom` provider (see hermes_cli/web_server.py
 * `_apply_main_model_assignment` + the local-endpoint BYOK flow), so this is the
 * exact shape that routes inference through the relay.
 *
 * `base_url` and `model` come from the provision-key response when available
 * (server is the source of truth); the env/DEFAULT_* fallback is only for the
 * boot seed before we hold a response. Throws on a missing key — a managed block
 * without a credential would 401 every request, which is worse than falling back
 * to BYOK; callers gate on `resolveManagedRelayCredential` first.
 *
 * @param {string} relayKey  the user's relay-valid cloud key
 * @param {Record<string, string | undefined>} [env]
 * @param {{ baseUrl?: string, model?: string }} [overrides] from provision-key
 * @returns {{ default: string, provider: string, base_url: string, api_key: string }}
 */
function buildManagedModelConfig(relayKey, env = {}, overrides = {}) {
  const key = String(relayKey || '').trim()
  if (!key) {
    throw new Error('buildManagedModelConfig: a relay key is required.')
  }
  const endpoints = resolveApexEndpoints(env)
  return {
    default: String(overrides.model || '').trim() || endpoints.model,
    provider: MANAGED_PROVIDER,
    base_url: trimTrailingSlash(overrides.baseUrl || '') || endpoints.relayBaseUrl,
    api_key: key
  }
}

/**
 * Parse + validate the provision-key response into the fields the desktop
 * persists and applies: { apiKey, baseUrl, model }. Returns null when the key is
 * missing (caller falls back to BYOK). base_url/model fall back to env defaults
 * if the server omits them, but the server is expected to send both.
 *
 * @param {unknown} body  response of POST /api/v1/desktop/provision-key
 * @param {Record<string, string | undefined>} [env]
 * @returns {{ apiKey: string, baseUrl: string, model: string } | null}
 */
function parseProvisionResponse(body, env = {}) {
  const key = relayKeyFromResponse(body)
  if (!key) return null
  const endpoints = resolveApexEndpoints(env)
  const baseUrl =
    body && typeof body === 'object' && typeof body.base_url === 'string' ? trimTrailingSlash(body.base_url) : ''
  const model = body && typeof body === 'object' && typeof body.model === 'string' ? body.model.trim() : ''
  return {
    apiKey: key,
    baseUrl: baseUrl || endpoints.relayBaseUrl,
    model: model || endpoints.model
  }
}

/**
 * Serialize the managed `model:` block to a YAML snippet for seedDefaultModelConfig.
 * Hand-rolled (no yaml dep — this module is dependency-free like its siblings);
 * values are simple scalars (URL, slug, opaque key) with no YAML-special chars,
 * but we still quote the key defensively since it is opaque input.
 *
 * @param {{ default: string, provider: string, base_url: string, api_key: string }} block
 * @returns {string}
 */
function managedModelConfigYaml(block) {
  const q = v => JSON.stringify(String(v)) // JSON string == valid YAML double-quoted scalar
  return (
    'model:\n' +
    `  default: ${block.default}\n` +
    `  provider: ${block.provider}\n` +
    `  base_url: ${q(block.base_url)}\n` +
    `  api_key: ${q(block.api_key)}\n`
  )
}

/**
 * Classify a credential-resolution result. The desktop should only switch the
 * default to managed when we actually hold a relay key; otherwise it must fall
 * back to BYOK. Centralizing the rule avoids each call site re-deriving it.
 *
 * @param {{ enabled: boolean, key: string | null }} state
 * @returns {'managed' | 'byok'}
 */
function defaultModelPath(state) {
  return state && state.enabled && typeof state.key === 'string' && state.key.trim() ? 'managed' : 'byok'
}

/**
 * Extract a relay key from the (future) relay-key endpoint response, tolerating
 * the couple of shapes the backend might return. Returns null when none present
 * so the caller falls back to BYOK rather than seeding an empty key.
 *
 * @param {unknown} body
 * @returns {string | null}
 */
function relayKeyFromResponse(body) {
  if (!body || typeof body !== 'object') return null
  const candidate =
    body.relay_key ?? body.api_key ?? body.key ?? (body.item && (body.item.key ?? body.item.api_key))
  const key = typeof candidate === 'string' ? candidate.trim() : ''
  return key || null
}

/**
 * Extract a JWT access token from the login response.
 * Mirrors the cloud auth route shape: `{ access_token, token_type: 'bearer' }`.
 *
 * @param {unknown} body
 * @returns {string | null}
 */
function accessTokenFromLogin(body) {
  if (!body || typeof body !== 'object') return null
  const token = typeof body.access_token === 'string' ? body.access_token.trim() : ''
  return token || null
}

module.exports = {
  DEFAULT_AUTH_BASE,
  DEFAULT_API_BASE,
  DEFAULT_RELAY_BASE_URL,
  DEFAULT_MANAGED_MODEL,
  MANAGED_MODEL_DISPLAY,
  MANAGED_PROVIDER,
  LOGIN_PATH,
  REGISTER_PATH,
  PROVISION_KEY_PATH,
  GOOGLE_START_PATH,
  WEB_LOGIN_PATH,
  accessTokenFromLogin,
  apexWebLoginUrl,
  buildManagedModelConfig,
  defaultModelPath,
  googleStartUrl,
  isLoopbackUrl,
  isManagedEnabled,
  managedModelConfigYaml,
  parseLoopbackCallback,
  parseProvisionResponse,
  relayKeyFromResponse,
  resolveApexEndpoints
}
