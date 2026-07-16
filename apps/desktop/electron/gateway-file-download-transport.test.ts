/**
 * Wiring coverage for the main.ts gateway download transports. These functions
 * pull in main-process singletons (https/http, electronNet, the OAuth session,
 * the save dialog), so we assert on their source shape — the same approach as
 * oauth-session-request.test.ts — while gateway-file-download.test.ts unit-tests
 * the extracted streaming/decoding logic behaviorally.
 */

import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import test from 'node:test'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const source = fs.readFileSync(path.join(__dirname, 'main.ts'), 'utf8')

function extract(startMarker: string, endMarker: string): string {
  const start = source.indexOf(startMarker)
  assert.notEqual(start, -1, `${startMarker} should exist`)
  const end = source.indexOf(endMarker, start + startMarker.length)
  assert.notEqual(end, -1, `boundary after ${startMarker} should exist`)

  return source.slice(start, end)
}

test('token transport streams to disk instead of buffering the whole body', () => {
  const fn = extract('function downloadViaTokenToFile', '\nfunction ')

  // Delegates byte-moving to the streaming finalizer...
  assert.match(fn, /finalizeGatewayDownload\(/)
  // ...and must NOT accumulate the full response before writing.
  assert.doesNotMatch(fn, /Buffer\.concat/)
  assert.doesNotMatch(fn, /chunks\.push/)
  // Idle timeout is dropped once headers arrive so the dialog/stream isn't killed.
  assert.match(fn, /setTimeout\(0\)/)
})

test('oauth transport streams to disk instead of buffering the whole body', () => {
  const fn = extract('function downloadViaOauthSessionToFile', '\nasync function finalizeGatewayDownload')

  assert.match(fn, /electronNet\.request/)
  assert.match(fn, /finalizeGatewayDownload\(/)
  assert.doesNotMatch(fn, /Buffer\.concat/)
  assert.doesNotMatch(fn, /chunks\.push/)
})

test('finalizeGatewayDownload prompts a save dialog then streams the response', () => {
  const fn = extract('async function finalizeGatewayDownload', '\nfunction readGatewayErrorText')

  assert.match(fn, /dialog\.showSaveDialog/)
  assert.match(fn, /pumpStreamToFile\(/)
  // HTTP errors carry their status so a 404 can trigger the fallback.
  assert.match(fn, /error\.statusCode = statusCode/)
})

test('saveGatewayFile falls back to the data-url route only on 404', () => {
  const fn = extract('async function saveGatewayFile', '\nasync function saveGatewayFileViaDataUrl')

  assert.match(fn, /\/api\/fs\/download\?path=/)
  assert.match(fn, /isNotFoundError\(error\)/)
  assert.match(fn, /saveGatewayFileViaDataUrl\(/)
})

test('data-url fallback reads the capped route and decodes it', () => {
  const fn = extract('async function saveGatewayFileViaDataUrl', '// Mint a single-use WS ticket')

  assert.match(fn, /\/api\/fs\/read-data-url\?path=/)
  assert.match(fn, /parseDataUrlToBuffer\(/)
})
