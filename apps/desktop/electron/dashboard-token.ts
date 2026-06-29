/**
 * Helpers for local dashboard session-token discovery.
 *
 * The desktop main process can pass HERMES_DASHBOARD_SESSION_TOKEN when it
 * spawns the local dashboard, but the dashboard is the source of truth for the
 * token it actually serves to the renderer. If those drift, HTTP readiness
 * probes still pass while /api/ws rejects the renderer's token.
 */

const DEFAULT_TOKEN_FETCH_TIMEOUT_MS = 3_000

async function fetchPublicText(url, options: any = {}) {
  const { protocol } = new URL(url)

  if (protocol !== 'http:' && protocol !== 'https:') {
    throw new Error(`Unsupported Hermes backend URL protocol: ${protocol}`)
  }

  const timeoutMs = options.timeoutMs ?? DEFAULT_TOKEN_FETCH_TIMEOUT_MS

  const res = await fetch(url, { signal: AbortSignal.timeout(timeoutMs) }).catch(error => {
    if (error.name === 'TimeoutError') {
      throw new Error(`Timed out connecting to Hermes backend after ${timeoutMs}ms`)
    }

    throw error
  })

  const text = await res.text()

  if (!res.ok) {throw new Error(`${res.status}: ${text || res.statusText}`)}

  return text
}

function extractInjectedDashboardToken(html) {
  const match = /window\.__HERMES_SESSION_TOKEN__\s*=\s*("(?:\\.|[^"\\])*")/.exec(String(html || ''))

  if (!match) {return null}

  try {
    return JSON.parse(match[1])
  } catch {
    return null
  }
}

function dashboardIndexUrl(baseUrl) {
  return `${String(baseUrl || '').replace(/\/+$/, '')}/`
}

async function resolveServedDashboardToken(baseUrl, fallbackToken, options: any = {}) {
  const fetchText = options.fetchText || fetchPublicText

  const html = await fetchText(dashboardIndexUrl(baseUrl), {
    timeoutMs: options.timeoutMs ?? DEFAULT_TOKEN_FETCH_TIMEOUT_MS
  })

  const servedToken = extractInjectedDashboardToken(html)

  if (servedToken && servedToken !== fallbackToken && typeof options.rememberLog === 'function') {
    options.rememberLog('[boot] dashboard served a different session token; using served token for WebSocket auth')
  }

  return servedToken || fallbackToken
}

/**
 * A served token that differs from our spawn token while our child is DEAD
 * came from a process we did not spawn (orphan/port squatter that satisfied
 * the public /api/status readiness probe). With a live child the mismatch is
 * benign: our own backend regenerated the token because the env pin did not
 * survive the spawn.
 */
function isForeignBackendToken({ servedToken, spawnToken, childAlive }) {
  return Boolean(servedToken) && servedToken !== spawnToken && !childAlive
}

/**
 * Resolve the token the backend actually serves, adopting benign drift and
 * failing loudly on a foreign backend. `childAlive` is a thunk so liveness is
 * sampled after the fetch, not before.
 */
async function adoptServedDashboardToken(baseUrl, spawnToken, { childAlive, label = 'Hermes backend', ...options }) {
  const servedToken = await resolveServedDashboardToken(baseUrl, spawnToken, options).catch(error => {
    options.rememberLog?.(`[boot] could not read served dashboard token (${label}): ${error.message}`)

    return spawnToken
  })

  if (isForeignBackendToken({ servedToken, spawnToken, childAlive: childAlive() })) {
    throw new Error(
      `${label} exited and ${dashboardIndexUrl(baseUrl)} is served by a process we did not spawn; refusing its session token.`
    )
  }

  return servedToken
}

export { adoptServedDashboardToken,
  dashboardIndexUrl,
  DEFAULT_TOKEN_FETCH_TIMEOUT_MS,
  extractInjectedDashboardToken,
  fetchPublicText,
  isForeignBackendToken,
  resolveServedDashboardToken }
