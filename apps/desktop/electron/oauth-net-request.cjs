/**
 * Helpers for Electron net.request calls that ride the OAuth session partition.
 *
 * Electron's ClientRequest forbids app-set restricted headers such as
 * Content-Length. Let Chromium frame the body itself; only set the JSON content
 * type here.
 */

function serializeJsonBody(body) {
  return body === undefined ? undefined : Buffer.from(JSON.stringify(body))
}

function setJsonRequestHeaders(request, headers = {}) {
  request.setHeader('Content-Type', 'application/json')
  for (const [name, value] of Object.entries(headers)) {
    if (value === undefined || value === null || name.toLowerCase() === 'content-length') continue
    request.setHeader(name, value)
  }
}

function parseOauthRequestUrl(url) {
  let parsed
  try {
    parsed = new URL(url)
  } catch (error) {
    throw new Error(`Invalid URL: ${error.message}`)
  }
  if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
    throw new Error(`Unsupported Hermes backend URL protocol: ${parsed.protocol}`)
  }
  return parsed
}

function readResponseText(chunks) {
  return Buffer.concat(chunks.map((chunk) => Buffer.from(chunk))).toString('utf8')
}

function createHttpError(statusCode, text) {
  const err = new Error(`${statusCode}: ${text || ''}`)
  err.statusCode = statusCode
  return err
}

function ensureJsonResponse(url, response, text) {
  const statusCode = response.statusCode || 500
  if (statusCode >= 400) throw createHttpError(statusCode, text)
  if (!text) return null

  const looksHtml = /^\s*<(?:!doctype|html)/i.test(text)
  const contentType = String(response.headers['content-type'] || response.headers['Content-Type'] || '')
  if (looksHtml || contentType.includes('text/html')) {
    throw new Error(`Expected JSON from ${url} but got HTML (status ${statusCode}).`)
  }
  try {
    return JSON.parse(text)
  } catch {
    throw new Error(`Invalid JSON from ${url} (status ${statusCode}): ${text.slice(0, 200)}`)
  }
}

module.exports = {
  createHttpError,
  ensureJsonResponse,
  parseOauthRequestUrl,
  readResponseText,
  serializeJsonBody,
  setJsonRequestHeaders
}
