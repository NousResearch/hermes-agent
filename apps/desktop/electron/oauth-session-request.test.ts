/**
 * Regression coverage for the OAuth-session Electron net.request path.
 *
 * Electron net rejects manual Content-Length/Host headers with
 * net::ERR_INVALID_ARGUMENT. Node HTTP helpers may still set Content-Length;
 * this guard is scoped to fetchJsonViaOauthSession only.
 */

import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import { test } from 'vitest'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const source = fs.readFileSync(path.join(__dirname, 'main.ts'), 'utf8')

function extractFetchJsonViaOauthSession() {
  const start = source.indexOf('function fetchJsonViaOauthSession')
  const end = source.indexOf('// Mint a single-use WS ticket', start)
  assert.notEqual(start, -1, 'fetchJsonViaOauthSession should exist')
  assert.notEqual(end, -1, 'fetchJsonViaOauthSession boundary should exist')

  return source.slice(start, end)
}

test('OAuth Electron net request does not set forbidden Content-Length header', () => {
  const fn = extractFetchJsonViaOauthSession()

  assert.match(fn, /electronNet\.request/)
  assert.doesNotMatch(fn, /setHeader\((['"])Content-Length\1/)
  assert.match(fn, /request\.write\(body\)/)
})

test('OAuth Electron net request handles truncated response errors', () => {
  const fn = extractFetchJsonViaOauthSession()

  assert.match(fn, /res\.on\(['"]error['"], error =>/)
  assert.match(fn, /if \(timedOut\)/)
})

test('remote revalidation uses Electron networking for OS-trusted CAs', () => {
  const start = source.indexOf("ipcMain.handle('hermes:connection:revalidate'")
  const end = source.indexOf("ipcMain.handle('hermes:backend:touch'", start)
  assert.notEqual(start, -1, 'revalidation handler should exist')
  assert.notEqual(end, -1, 'revalidation handler boundary should exist')

  // Strip line comments first: this test reasons about the CODE in the handler,
  // and the comment explaining *why* we avoid the Node-https probe legitimately
  // names it. Matching raw source would fail on the explanation itself.
  const handler = source.slice(start, end).replace(/^[^\S\n]*\/\/.*$/gm, '')
  assert.match(handler, /fetchJsonViaOauthSession/)
  assert.doesNotMatch(handler, /fetchPublicJson/)
})
