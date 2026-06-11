/**
 * Helpers for local dashboard session-token discovery.
 *
 * The desktop main process can pass HERMES_DASHBOARD_SESSION_TOKEN when it
 * spawns the local dashboard, but the dashboard is the source of truth for the
 * token it actually serves to the renderer. If those drift, HTTP readiness
 * probes still pass while /api/ws rejects the renderer's token.
 */

const http = require('node:http')
const https = require('node:https')

const DEFAULT_TOKEN_FETCH_TIMEOUT_MS = 3_000

function fetchPublicText(url, options = {}) {
  return new Promise((resolve, reject) => {
    let parsed
    try {
      parsed = new URL(url)
    } catch (error) {
      reject(new Error(`Invalid URL: ${error.message}`))
      return
    }

    if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
      reject(new Error(`Unsupported Hermes backend URL protocol: ${parsed.protocol}`))
      return
    }

    const client = parsed.protocol === 'https:' ? https : http
    const timeoutMs = options.timeoutMs ?? DEFAULT_TOKEN_FETCH_TIMEOUT_MS
    const req = client.request(parsed, { method: options.method || 'GET' }, res => {
      const chunks = []
      res.on('data', chunk => chunks.push(chunk))
      res.on('end', () => {
        const text = Buffer.concat(chunks).toString('utf8')
        if ((res.statusCode || 500) >= 400) {
          reject(new Error(`${res.statusCode}: ${text || res.statusMessage}`))
          return
        }
        resolve(text)
      })
    })

    req.on('error', reject)
    req.setTimeout(timeoutMs, () => {
      req.destroy(new Error(`Timed out connecting to Hermes backend after ${timeoutMs}ms`))
    })
    req.end()
  })
}

function extractInjectedDashboardToken(html) {
  const match = /window\.__HERMES_SESSION_TOKEN__\s*=\s*("(?:\\.|[^"\\])*")/.exec(String(html || ''))
  if (!match) return null
  try {
    return JSON.parse(match[1])
  } catch {
    return null
  }
}

function dashboardIndexUrl(baseUrl) {
  return `${String(baseUrl || '').replace(/\/+$/, '')}/`
}

async function resolveServedDashboardToken(baseUrl, fallbackToken, options = {}) {
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
 * Decide whether a served-token mismatch means we are talking to a backend we
 * did NOT spawn.
 *
 * The desktop pins HERMES_DASHBOARD_SESSION_TOKEN on every backend it spawns,
 * and the dashboard honors that env at import — so a LIVE child of ours always
 * serves our token. The only way the served token differs while our child is
 * dead is that the readiness probe (public /api/status) answered from a
 * different process: an orphaned dashboard or port squatter that won the bind
 * race while our child exited. Adopting that process's token would silently
 * authenticate the renderer against a foreign backend (possibly the wrong
 * profile), so callers must fail loudly instead.
 *
 * A mismatch with a live child is the benign case the served-token fallback
 * exists for: our own child served a regenerated token because the env pin
 * did not survive the spawn (e.g. shell-wrapped CLI shims).
 */
function isForeignBackendToken({ servedToken, spawnToken, childAlive }) {
  return Boolean(servedToken) && servedToken !== spawnToken && !childAlive
}

module.exports = {
  DEFAULT_TOKEN_FETCH_TIMEOUT_MS,
  dashboardIndexUrl,
  extractInjectedDashboardToken,
  fetchPublicText,
  isForeignBackendToken,
  resolveServedDashboardToken
}
