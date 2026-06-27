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

// Endpoint paths. LOGIN_PATH is on AUTH_BASE; PROVISION_KEY_PATH is on API_BASE.
const LOGIN_PATH = '/api/v1/auth/login'
const PROVISION_KEY_PATH = '/api/v1/desktop/provision-key'

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
    provisionKeyUrl: `${apiBase}${PROVISION_KEY_PATH}`
  }
}

/**
 * True when the managed-LLM default path is enabled for this install. Single
 * gate the rest of the app (boot seed, onboarding, IPC) consults so the flag
 * check isn't scattered.
 *
 * Shipped OFF by default as a safety gate while the backend provision-key
 * endpoint rolls out to prod: with it off, the desktop behaves exactly like the
 * current BYOK build (zero regression). Flip ON — set `APEXNODES_MANAGED=1`, or
 * change this default to `raw !== '0'` — the moment the endpoint is live, and
 * managed becomes the zero-key default. Even when ON, every path auto-degrades
 * to BYOK if provision-key is unreachable, so flipping it early only changes the
 * first-run UI (managed sign-in panel first), never breaks chat.
 *
 * @param {Record<string, string | undefined>} [env]
 */
function isManagedEnabled(env = {}) {
  const raw = String(env.APEXNODES_MANAGED ?? '').trim().toLowerCase()
  return raw === '1' || raw === 'true' || raw === 'yes' || raw === 'on'
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
  PROVISION_KEY_PATH,
  accessTokenFromLogin,
  buildManagedModelConfig,
  defaultModelPath,
  isManagedEnabled,
  managedModelConfigYaml,
  parseProvisionResponse,
  relayKeyFromResponse,
  resolveApexEndpoints
}
