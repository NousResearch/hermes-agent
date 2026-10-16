/**
 * Helpers for Electron net.request calls that ride the OAuth session partition.
 *
 * Electron's ClientRequest forbids app-set restricted headers such as
 * Content-Length. Let Chromium frame the body itself; set only the JSON content
 * type and, for state-changing methods, the Origin Chromium does not synthesize
 * for main-process requests.
 */

function serializeJsonBody(body) {
  return body === undefined ? undefined : Buffer.from(JSON.stringify(body))
}

const SAFE_METHODS = new Set(['GET', 'HEAD', 'OPTIONS'])

function setJsonRequestHeaders(request, options: { method?: string; url?: string | URL } = {}) {
  request.setHeader('Content-Type', 'application/json')

  const method = String(options.method || 'GET').toUpperCase()

  // Electron does not synthesize an Origin for main-process net.request calls, so a gateway that
  // enforces same-origin on state-changing requests rejected every Desktop POST/PATCH/DELETE.
  if (!SAFE_METHODS.has(method) && options.url) {
    request.setHeader('Origin', new URL(options.url).origin)
  }
}

export { serializeJsonBody, setJsonRequestHeaders }
