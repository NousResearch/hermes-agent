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

function setJsonRequestHeaders(request) {
  request.setHeader('Content-Type', 'application/json')
}

module.exports = {
  serializeJsonBody,
  setJsonRequestHeaders
}
