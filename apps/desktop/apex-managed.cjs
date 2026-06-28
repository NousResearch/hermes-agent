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
// model from the `model` field the runtime sends — the relay routes by DB truth
// (verified: the relay ignores the request's `model` entirely and returns
// `deepseek-v4-pro` for ANY value, including unknown ids), so the model id we
// write to config is cosmetic to the relay.
const DEFAULT_MANAGED_MODEL = 'deepseek-v4-pro'

// The model id we actually WRITE to config.yaml (`model.default` + the
// custom_providers entry's `model`) and show in the UI.
//
// ⚠️ This must be a name that is NOT an exact id in any built-in provider's
// static model catalog (`hermes_cli/models.py` `_PROVIDER_MODELS`). The bare
// routed id `deepseek-v4-pro` IS in the built-in DeepSeek catalog, and that is
// exactly what broke managed chat:
//
//   The desktop runs `hermes dashboard`; its embedded chat builds the agent via
//   `tui_gateway/server.py::_make_agent`. At boot (no per-session override) that
//   path resolves the model through `_resolve_startup_runtime`, which — when an
//   inference-model env hint is present (`HERMES_MODEL`/`HERMES_INFERENCE_MODEL`,
//   set by the runtime's own launcher and inheritable into the backend) — runs
//   `detect_static_provider_for_model(<model id>, …)`. That does an EXACT match
//   against the built-in catalogs (`models.py:1885`), so `deepseek-v4-pro`
//   resolves to provider `deepseek` (and `kimi-k2.6`→`kimi-coding`,
//   `glm-5.2`→`zai`), OVERRIDING the configured `provider: custom`. The built-in
//   provider has no key → `agent/agent_init.py` raises "Provider 'deepseek' is
//   set in config.yaml but no API key was found." The gateway caches that failed
//   build (`agent_build_started`), so switching models in the picker can't
//   recover the session — every selection shows the same sticky boot error. The
//   ONLY fix is a boot config whose model id does not collide.
//
// `deepseek-v4-pro-APEX` (the ApexNodes display name) is collision-free
// (`detect_static_provider_for_model` returns None — verified) AND relay-valid
// (HTTP 200, routed to deepseek-v4-pro — verified). Using it as the config model
// id makes the startup path resolve to the relay in every case (with or without
// the env hint), proven against the runtime venv.
const MANAGED_MODEL_DISPLAY = 'deepseek-v4-pro-APEX'

// The runtime treats the relay as a generic OpenAI-compatible endpoint, so the
// provider slug is the same `custom` the local/custom BYOK flow uses. Reusing
// `custom` means zero new runtime provider plumbing.
const MANAGED_PROVIDER = 'custom'

// Display name of the relay's `custom_providers:` entry. The runtime groups
// custom endpoints by this name in its model picker (users see an
// "APEX-NODES.COM" group), and Hermes' own writer
// (`hermes_cli/main.py::_save_custom_provider`) uses the exact same
// `{name, base_url, api_key, model}` entry shape. We register this entry so the
// relay is a *named* custom provider — the format Hermes produces after a
// `hermes model` custom-endpoint selection — which keeps the endpoint durable
// across `/model` picker switches and session resume (those persist
// `provider: custom:<slug>`, which only resolves when the named entry exists).
// The collision fix itself is the non-colliding model id above; this entry is
// the native-format hardening that goes with it.
const MANAGED_PROVIDER_NAME = 'Apex-nodes.com'

// ── ApexNodes China default profile (hc-392) ───────────────────────────────
// The desktop pre-seeds config.yaml BEFORE install.sh can copy
// cli-config.yaml.example (seedDefaultModelConfig in main.cjs only writes when
// config.yaml is absent, and install.sh's example-copy is likewise
// absent-gated — so the seed wins). The China profile therefore CANNOT rely on
// cli-config.yaml.example reaching the desktop; we fold the same two policy
// lists into the seed here so skill-cut + Copilot-disable actually take effect
// on a fresh desktop install. The runtime reads these from config.yaml via
// agent.skill_utils.get_disabled_skill_names() and
// hermes_cli.model_switch.list_authenticated_providers (model.disabled_providers).
//
// Providers never probed / live-fetched / shown. Matched case-insensitively
// against the Hermes slug + its models.dev id.
const MODEL_DISABLED_PROVIDERS = ['copilot']

// Skills physically present in ~/.hermes/skills/ but kept INACTIVE by default
// (never loaded until removed from this list in Settings → Skills). We disable
// rather than delete so upstream merges stay painless. Names below are the
// SKILL.md frontmatter `name:` (which the toggle matches) — note the four that
// differ from their folder names (serving-llms-vllm, evaluating-llms-harness,
// segment-anything-model, audiocraft-audio-generation). MUST stay in sync with
// the `skills.disabled` list in cli-config.yaml.example (the pure-CLI path).
const SEED_DISABLED_SKILLS = [
  // GROUP A — China / product focus (墙外 / 国内不稳 / 竞品 dev)
  'google-workspace', 'xurl', 'youtube-content', 'polymarket',
  'teams-meeting-pipeline', 'claude-code', 'codex', 'opencode',
  'notion', 'airtable', 'gif-search', 'arxiv',
  'github-auth', 'github-code-review', 'github-issues',
  'github-pr-workflow', 'github-repo-management',
  // GROUP B — capable but niche (toggle on per-need): dev
  'codebase-inspection', 'simplify-code', 'test-driven-development',
  'systematic-debugging', 'requesting-code-review', 'node-inspect-debugger',
  'python-debugpy', 'jupyter-live-kernel', 'himalaya', 'design-md',
  'hermes-agent', 'hermes-agent-skill-authoring',
  // ML / research
  'serving-llms-vllm', 'llama-cpp', 'weights-and-biases',
  'evaluating-llms-harness', 'research-paper-writing', 'pretext',
  'segment-anything-model', 'comfyui', 'audiocraft-audio-generation',
  // creative / 小众
  'heartmula', 'songwriting-and-ai-music', 'songsee', 'manim-video',
  'p5js', 'touchdesigner-mcp', 'openhue',
  // internal / QA
  'kanban-orchestrator', 'kanban-worker', 'dogfood', 'spike'
]

/**
 * Render the `model.disabled_providers` YAML lines (indented to sit INSIDE the
 * `model:` block). Returns '' when the list is empty. Kept as a helper so the
 * managed (apex-managed) and BYOK (main.cjs raw string) seed paths emit the
 * identical block.
 *
 * @param {string[]} [providers]
 * @returns {string}
 */
function modelDisabledProvidersYaml(providers = MODEL_DISABLED_PROVIDERS) {
  const list = Array.isArray(providers) ? providers.filter(p => String(p || '').trim()) : []
  if (!list.length) return ''
  let yaml = '  disabled_providers:\n'
  for (const p of list) yaml += `    - ${String(p).trim()}\n`
  return yaml
}

/**
 * Render the top-level `skills.disabled` YAML block. Returns '' when empty.
 * Top-level `skills:` key — no collision with the `model:` / `display:` blocks
 * the seed already emits.
 *
 * @param {string[]} [skills]
 * @returns {string}
 */
function seedSkillsBlockYaml(skills = SEED_DISABLED_SKILLS) {
  const list = Array.isArray(skills) ? skills.filter(s => String(s || '').trim()) : []
  if (!list.length) return ''
  let yaml =
    '# ApexNodes China default profile (hc-392): skills shipped but OFF by\n' +
    '# default. Toggle any on in Settings → Skills. Files are kept (not\n' +
    '# deleted) so upstream merges stay clean.\n' +
    'skills:\n' +
    '  disabled:\n'
  for (const s of list) yaml += `    - ${String(s).trim()}\n`
  return yaml
}

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
  // The display id is what gets WRITTEN to config (collision-free with built-in
  // catalogs — see MANAGED_MODEL_DISPLAY). Precedence:
  //   1. explicit APEXNODES_MANAGED_MODEL_DISPLAY override
  //   2. when only APEXNODES_MANAGED_MODEL is overridden (e.g. staging), derive a
  //      collision-free id by appending the `-APEX` brand suffix to it
  //   3. the prod default display name
  const explicitDisplay = String(env.APEXNODES_MANAGED_MODEL_DISPLAY || '').trim()
  const modelDisplay =
    explicitDisplay ||
    (env.APEXNODES_MANAGED_MODEL ? `${model}-APEX` : MANAGED_MODEL_DISPLAY)

  return {
    authBase,
    apiBase,
    relayBaseUrl,
    model,
    modelDisplay,
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
 * The written model id is the ApexNodes display name (`MANAGED_MODEL_DISPLAY`,
 * env-overridable), NOT the raw routed id — the raw `deepseek-v4-pro` collides
 * with the built-in DeepSeek catalog and gets the agent init mis-routed to the
 * keyless built-in `deepseek` provider (see MANAGED_MODEL_DISPLAY above). The
 * relay routes by DB truth and ignores this id, so it is safe + cosmetic on the
 * wire; locally it is the collision-free anchor that keeps resolution on the
 * relay's custom endpoint.
 *
 * Also returns a `custom_providers` entry registering the relay as a named
 * custom provider (`{name, base_url, api_key, model}` — Hermes' native shape).
 * The `model:` block keeps `provider: custom` + the relay `base_url`/`api_key`
 * (so the resolved provider class matches and there is no per-turn re-switch),
 * while the registered entry keeps the endpoint durable across picker switches /
 * session resume (which persist `provider: custom:<slug>`).
 *
 * @param {string} relayKey  the user's relay-valid cloud key
 * @param {Record<string, string | undefined>} [env]
 * @param {{ baseUrl?: string, model?: string }} [overrides] from provision-key
 * @returns {{
 *   default: string, provider: string, base_url: string, api_key: string,
 *   custom_providers: Array<{ name: string, base_url: string, api_key: string, model: string }>
 * }}
 */
function buildManagedModelConfig(relayKey, env = {}, overrides = {}) {
  const key = String(relayKey || '').trim()
  if (!key) {
    throw new Error('buildManagedModelConfig: a relay key is required.')
  }
  const endpoints = resolveApexEndpoints(env)
  // The model id WRITTEN to config must be collision-free with the built-in
  // catalogs (see MANAGED_MODEL_DISPLAY). The relay ignores the model id (routes
  // by DB truth), so a provision-key `overrides.model` is only honored when it is
  // ALREADY a non-colliding ApexNodes display id (ends with the `-APEX` brand
  // suffix); otherwise we use the display name so a raw routed id like
  // `deepseek-v4-pro` can never re-seed the collision the next time config is
  // (re)written at boot.
  const overrideModel = String(overrides.model || '').trim()
  const model = /-APEX$/i.test(overrideModel) ? overrideModel : endpoints.modelDisplay
  const baseUrl = trimTrailingSlash(overrides.baseUrl || '') || endpoints.relayBaseUrl
  return {
    default: model,
    provider: MANAGED_PROVIDER,
    base_url: baseUrl,
    api_key: key,
    // Register the relay as a named custom provider (Hermes-native shape) so the
    // endpoint stays durable across picker switches / resume. Same id as
    // model.default so both anchors agree.
    custom_providers: [
      {
        name: MANAGED_PROVIDER_NAME,
        base_url: baseUrl,
        api_key: key,
        model
      }
    ]
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
 * Serialize the managed `model:` block (and the `custom_providers:` entry that
 * registers the relay) to a YAML snippet for seedDefaultModelConfig. Hand-rolled
 * (no yaml dep — this module is dependency-free like its siblings); values are
 * simple scalars (URL, slug, opaque key), but we double-quote the URL/key/name
 * defensively since they are opaque/external input.
 *
 * @param {{
 *   default: string, provider: string, base_url: string, api_key: string,
 *   custom_providers?: Array<{ name: string, base_url: string, api_key: string, model: string }>
 * }} block
 * @param {{ disabledProviders?: string[] }} [opts]  hc-392: when
 *   `disabledProviders` is given, its `disabled_providers:` lines are emitted
 *   INSIDE this `model:` block (a second top-level `model:` block would be a
 *   duplicate YAML key).
 * @returns {string}
 */
function managedModelConfigYaml(block, opts = {}) {
  const q = v => JSON.stringify(String(v)) // JSON string == valid YAML double-quoted scalar
  let yaml =
    'model:\n' +
    `  default: ${block.default}\n` +
    `  provider: ${block.provider}\n` +
    `  base_url: ${q(block.base_url)}\n` +
    `  api_key: ${q(block.api_key)}\n`
  if (opts && opts.disabledProviders) {
    yaml += modelDisabledProvidersYaml(opts.disabledProviders)
  }
  const entries = Array.isArray(block.custom_providers) ? block.custom_providers : []
  if (entries.length) {
    yaml += 'custom_providers:\n'
    for (const entry of entries) {
      yaml +=
        `  - name: ${q(entry.name)}\n` +
        `    base_url: ${q(entry.base_url)}\n` +
        `    api_key: ${q(entry.api_key)}\n` +
        `    model: ${entry.model}\n`
    }
  }
  return yaml
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
  MANAGED_PROVIDER_NAME,
  MODEL_DISABLED_PROVIDERS,
  SEED_DISABLED_SKILLS,
  modelDisabledProvidersYaml,
  seedSkillsBlockYaml,
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
