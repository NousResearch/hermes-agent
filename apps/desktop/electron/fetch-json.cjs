/**
 * fetch-json.cjs
 *
 * HTTP JSON helper for the desktop Electron main process. Pulled out of
 * main.cjs so it's directly unit-testable with `node --test` and so the
 * password-auth header-bug (#63707) has a verifiable contract.
 *
 * The contract for the `X-Hermes-Session-Token` header:
 *   - When `token` is a non-empty string, set the header to that value.
 *   - When `token` is null/undefined/empty, DO NOT set the header at all
 *     (this is the password-auth case — the gateway authenticates via an
 *     HttpOnly session cookie held in the cookie jar, not a token header).
 *
 * Setting the header to `undefined` (the pre-#63707 bug) makes Node's
 * `http.request` throw ERR_INVALID_ARG_TYPE before anything hits the
 * network, surfacing to the renderer as `net::ERR_INVALID_ARGUMENT`
 * for every REST mutation (delete, archive, rename).
 */

const http = require('node:http')
const https = require('node:https')
const { DEFAULT_FETCH_TIMEOUT_MS, resolveTimeoutMs } = require('./hardening.cjs')

/**
 * Build the headers for a fetchJson request. Exported separately so the
 * header contract — and especially the "no token → no header" rule — can
 * be unit-tested without spinning up an HTTP server.
 *
 * @param {string|undefined|null} token  The session token, or nullish for
 *   cookie-based auth (e.g. password or OAuth modes).
 * @param {Buffer|undefined} body  Request body, when present, so we can
 *   add Content-Length.
 * @returns {Record<string, string>}  Headers ready for http.request.
 */
function fetchJsonHeaders(token, body) {
  const headers = {
    'Content-Type': 'application/json',
    ...(token ? { 'X-Hermes-Session-Token': token } : {}),
    ...(body ? { 'Content-Length': String(body.length) } : {})
  }
  return headers
}

/**
 * Issue an HTTP(S) JSON request and resolve with the parsed JSON body.
 * Throws on non-2xx status, HTML bodies (signals an unregistered /api
 * path falling through to the SPA index), and JSON parse failures.
 *
 * @param {string} url  Full http:// or https:// URL.
 * @param {string|undefined|null} [token]  Optional session-token header.
 * @param {object} [options]  { method, body, timeoutMs }.
 */
function fetchJson(url, token, options = {}) {
  return new Promise((resolve, reject) => {
    const body = options.body === undefined ? undefined : Buffer.from(JSON.stringify(options.body))
    const parsed = new URL(url)
    const client = parsed.protocol === 'https:' ? https : http
    const timeoutMs = resolveTimeoutMs(options.timeoutMs, DEFAULT_FETCH_TIMEOUT_MS)

    if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
      reject(new Error(`Unsupported Hermes backend URL protocol: ${parsed.protocol}`))
      return
    }

    const req = client.request(
      parsed,
      {
        method: options.method || 'GET',
        headers: fetchJsonHeaders(token, body)
      },
      res => {
        const chunks = []
        res.on('error', reject)
        res.on('data', chunk => chunks.push(chunk))
        res.on('end', () => {
          const text = Buffer.concat(chunks).toString('utf8')
          if ((res.statusCode || 500) >= 400) {
            reject(new Error(`${res.statusCode}: ${text || res.statusMessage}`))
            return
          }
          if (!text) {
            resolve(null)
            return
          }
          // A 2xx response whose body is HTML means the request fell through
          // to the SPA index.html (e.g. an unregistered /api path). JSON.parse
          // would throw an opaque `Unexpected token '<'` here, so surface a
          // clear diagnostic with the offending URL instead.
          const looksHtml = /^\s*<(?:!doctype|html)/i.test(text)
          const contentType = String(res.headers['content-type'] || '')
          if (looksHtml || contentType.includes('text/html')) {
            reject(
              new Error(
                `Expected JSON from ${url} but got HTML (status ${res.statusCode}). ` +
                  'The endpoint is likely missing on the Hermes backend.'
              )
            )
            return
          }
          try {
            resolve(JSON.parse(text))
          } catch {
            reject(new Error(`Invalid JSON from ${url} (status ${res.statusCode}): ${text.slice(0, 200)}`))
          }
        })
      }
    )

    req.on('error', reject)
    req.setTimeout(timeoutMs, () => {
      req.destroy(new Error(`Timed out connecting to Hermes backend after ${timeoutMs}ms`))
    })
    if (body) req.write(body)
    req.end()
  })
}

module.exports = {
  fetchJson,
  fetchJsonHeaders
}
