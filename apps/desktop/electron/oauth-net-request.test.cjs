/**
 * Tests for OAuth-session Electron net.request helpers.
 *
 * Run with: node --test electron/oauth-net-request.test.cjs
 */

const test = require('node:test')
const assert = require('node:assert/strict')

const {
  createHttpError,
  ensureJsonResponse,
  parseOauthRequestUrl,
  readResponseText,
  serializeJsonBody,
  setJsonRequestHeaders
} = require('./oauth-net-request.cjs')

test('serializeJsonBody returns undefined for absent bodies', () => {
  assert.equal(serializeJsonBody(undefined), undefined)
})

test('serializeJsonBody JSON-encodes request bodies', () => {
  const body = serializeJsonBody({ archived: true })
  assert.ok(Buffer.isBuffer(body))
  assert.equal(body.toString('utf8'), '{"archived":true}')
})

test('setJsonRequestHeaders does not set Electron-restricted Content-Length', () => {
  const headers = []
  const request = {
    setHeader(name, value) {
      headers.push([name, value])
    }
  }

  setJsonRequestHeaders(request, { Accept: 'application/json', 'Content-Length': '999' })

  assert.deepEqual(headers, [
    ['Content-Type', 'application/json'],
    ['Accept', 'application/json']
  ])
  assert.equal(
    headers.some(([name]) => name.toLowerCase() === 'content-length'),
    false
  )
})

test('parseOauthRequestUrl accepts http and https urls', () => {
  assert.equal(parseOauthRequestUrl('https://example.com/api').hostname, 'example.com')
  assert.equal(parseOauthRequestUrl('http://localhost:8642').protocol, 'http:')
})

test('parseOauthRequestUrl rejects unsupported protocols', () => {
  assert.throws(() => parseOauthRequestUrl('file:///tmp/nope'), /Unsupported Hermes backend URL protocol/)
})

test('readResponseText concatenates response chunks', () => {
  assert.equal(readResponseText(['{"ok":', 'true}']), '{"ok":true}')
})

test('createHttpError preserves status code', () => {
  const error = createHttpError(401, 'nope')
  assert.equal(error.statusCode, 401)
  assert.match(error.message, /^401: nope$/)
})

test('ensureJsonResponse returns null for empty body', () => {
  assert.equal(ensureJsonResponse('https://example.com', { statusCode: 204, headers: {} }, ''), null)
})

test('ensureJsonResponse parses json payloads', () => {
  assert.deepEqual(
    ensureJsonResponse('https://example.com', { statusCode: 200, headers: { 'content-type': 'application/json' } }, '{"ok":true}'),
    { ok: true }
  )
})

test('ensureJsonResponse rejects html responses', () => {
  assert.throws(
    () => ensureJsonResponse('https://example.com', { statusCode: 200, headers: { 'content-type': 'text/html' } }, '<html></html>'),
    /Expected JSON/
  )
})

test('ensureJsonResponse rejects invalid json payloads', () => {
  assert.throws(
    () => ensureJsonResponse('https://example.com', { statusCode: 200, headers: { 'content-type': 'application/json' } }, 'oops'),
    /Invalid JSON/
  )
})
