'use strict'

/**
 * apex-client-config.cjs
 *
 * Platform client-config sync — pure, dependency-free helpers (like the other
 * electron/*.cjs siblings) behind the versioned client config the APEX cloud
 * serves to every desktop. main.cjs wires the persistence, IPC and the boot /
 * post-sign-in refresh; the renderer (src/store/platform-config.ts) applies the
 * payload through the runtime's global-config dashboard API.
 *
 * BACKEND CONTRACT (built in parallel — tolerate absence)
 * -------------------------------------------------------
 *   GET {API_BASE}/api/v1/desktop/client-config          (PUBLIC, no auth)
 *     → { "version": <int>, "payload": {...}, "updated_at": "..." }
 *   GET …/client-config?known_version=<n>
 *     → may short-circuit to { "version": n, "unchanged": true }
 *   404 → no active config (treated as "nothing to sync", never an error).
 *
 * payload shape v1: { "config_yaml": { "<dotted.key>": <scalar>, … } } —
 * dotted keys into the runtime's global config. Unknown top-level payload
 * fields must be IGNORED (forward compat), so the parser preserves the payload
 * verbatim and consumers only read the fields they know.
 *
 * The synced state persists at userData/apex-client-config.json as
 *   { version, payload, fetchedAt, appliedVersion }
 * No secrets live in it, so it is plain JSON (no safeStorage encryption).
 * `version` is the newest payload fetched from the platform; `appliedVersion`
 * is the newest version the renderer finished applying — the gap between the
 * two is what drives a (re-)apply on the next gateway-open.
 */

const CLIENT_CONFIG_PATH = '/api/v1/desktop/client-config'

function trimTrailingSlash(value) {
  return String(value || '').replace(/\/+$/, '')
}

/**
 * Normalize a value that should be a config version: a positive integer.
 * Accepts an integer number or an all-digit string (lenient toward JSON
 * re-serialization); anything else → 0 ("no version").
 *
 * @param {unknown} value
 * @returns {number} positive integer, or 0 when the value is not one
 */
function normalizeVersion(value) {
  if (typeof value === 'number' && Number.isInteger(value) && value > 0) {
    return value
  }
  if (typeof value === 'string' && /^[0-9]+$/.test(value.trim())) {
    const parsed = Number.parseInt(value.trim(), 10)
    return Number.isSafeInteger(parsed) && parsed > 0 ? parsed : 0
  }
  return 0
}

/**
 * Build the client-config URL for an apiBase, optionally advertising the
 * version we already hold so the server can answer `{ unchanged: true }`.
 *
 * @param {string} apiBase e.g. https://api.apex-nodes.com
 * @param {number} [knownVersion] included as ?known_version=<n> when a
 *   positive integer
 * @returns {string}
 */
function clientConfigUrl(apiBase, knownVersion) {
  const base = trimTrailingSlash(apiBase)
  const known = normalizeVersion(knownVersion)
  const q = known > 0 ? `?known_version=${known}` : ''
  return `${base}${CLIENT_CONFIG_PATH}${q}`
}

/**
 * Validate + normalize a client-config response body. Returns null on garbage
 * so callers can treat any malformed body exactly like "no config" (fail-soft).
 *
 * Accepted shapes:
 *   { version: int>0, payload: object, updated_at?: string }  → full config
 *   { version: int>0, unchanged: true }                       → not modified
 *
 * Unknown top-level fields (on the body or inside payload) are ignored /
 * preserved verbatim — forward compat with future contract additions.
 *
 * @param {unknown} body parsed JSON response
 * @returns {null | { version: number, payload: Record<string, unknown> | null,
 *   unchanged: boolean, updatedAt: string | null }}
 */
function parseClientConfigResponse(body) {
  if (!body || typeof body !== 'object' || Array.isArray(body)) return null
  const version = normalizeVersion(body.version)
  if (!version) return null

  // Short-circuit shape: the server confirmed our known_version is current.
  // No payload accompanies it (and none is needed — we already hold it).
  if (body.unchanged === true) {
    return { version, payload: null, unchanged: true, updatedAt: null }
  }

  const payload = body.payload
  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) return null

  const updatedAt = typeof body.updated_at === 'string' && body.updated_at.trim() ? body.updated_at : null
  return { version, payload, unchanged: false, updatedAt }
}

/**
 * Should a fetched config version be applied (or stored) over what we already
 * have? True only when `fetchedVersion` is a positive integer strictly greater
 * than `appliedVersion` (garbage/absent appliedVersion counts as 0, so a first
 * ever config always applies; a version REGRESSION never does — the platform
 * bumps the version on every change, and re-applying an older payload over a
 * newer one would fight the server).
 *
 * @param {unknown} fetchedVersion
 * @param {unknown} appliedVersion
 * @returns {boolean}
 */
function shouldApply(fetchedVersion, appliedVersion) {
  const fetched = normalizeVersion(fetchedVersion)
  if (!fetched) return false
  return fetched > normalizeVersion(appliedVersion)
}

/**
 * Normalize the persisted apex-client-config.json content. Any garbage (missing
 * file parsed to undefined, corrupt JSON parsed to a scalar, tampered fields)
 * degrades to the empty state so boot can never throw over the cache.
 *
 * @param {unknown} raw parsed file content
 * @returns {{ version: number, payload: Record<string, unknown> | null,
 *   fetchedAt: number | null, appliedVersion: number }}
 */
function normalizeStoredClientConfig(raw) {
  const empty = { version: 0, payload: null, fetchedAt: null, appliedVersion: 0 }
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return empty
  const version = normalizeVersion(raw.version)
  const payload =
    raw.payload && typeof raw.payload === 'object' && !Array.isArray(raw.payload) ? raw.payload : null
  // A version without its payload is unusable — treat as empty so the next
  // fetch re-stores both.
  if (!version || !payload) return empty
  const fetchedAt = typeof raw.fetchedAt === 'number' && Number.isFinite(raw.fetchedAt) ? raw.fetchedAt : null
  return { version, payload, fetchedAt, appliedVersion: normalizeVersion(raw.appliedVersion) }
}

/**
 * Fetch + parse the platform client config. NEVER throws — returns null on any
 * failure (offline, 404 no-active-config, HTML, garbage body) so the boot /
 * post-sign-in refresh degrades to the cached state. Mirrors
 * apex-runtime-latest.cjs::resolveLatestRuntimePin: main.cjs passes its
 * fetchPublicJson as `fetchJson` (credential-free — this endpoint is PUBLIC and
 * must never see a token).
 *
 * @param {object} opts
 * @param {string} opts.apiBase resolveApexEndpoints(process.env).apiBase
 * @param {(url: string, options?: object) => Promise<any>} opts.fetchJson
 * @param {number} [opts.knownVersion] currently stored version (skip hint)
 * @param {number} [opts.timeoutMs]
 * @param {(msg: string) => void} [opts.log]
 * @returns {Promise<null | ReturnType<typeof parseClientConfigResponse>>}
 */
async function fetchClientConfig({ apiBase, fetchJson, knownVersion, timeoutMs = 5_000, log = () => {} }) {
  if (!apiBase || typeof fetchJson !== 'function') return null
  const url = clientConfigUrl(apiBase, knownVersion)
  let body
  try {
    body = await fetchJson(url, { timeoutMs })
  } catch (err) {
    // 404 (no active config), network error, HTML gateway page, timeout →
    // "nothing new"; the cached state stands.
    log(`[client-config] fetch unavailable (${(err && err.message) || err}); keeping cached state`)
    return null
  }
  const parsed = parseClientConfigResponse(body)
  if (!parsed) {
    log('[client-config] response body malformed; keeping cached state')
    return null
  }
  return parsed
}

module.exports = {
  CLIENT_CONFIG_PATH,
  clientConfigUrl,
  fetchClientConfig,
  normalizeStoredClientConfig,
  normalizeVersion,
  parseClientConfigResponse,
  shouldApply
}
