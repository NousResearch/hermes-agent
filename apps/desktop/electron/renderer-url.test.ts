import assert from 'node:assert/strict'
import path from 'node:path'
import { pathToFileURL } from 'node:url'

import { test } from 'vitest'

import { isRendererUrl } from './renderer-url'

const HTTP_BASE = 'http://127.0.0.1:47891'

test('isRendererUrl accepts the renderer origin itself', () => {
  assert.equal(isRendererUrl(`${HTTP_BASE}/`, HTTP_BASE), true)
  assert.equal(isRendererUrl(`${HTTP_BASE}/?win=secondary#/abc123`, HTTP_BASE), true)
})

// Regression: a `startsWith(base)` guard accepted this because everything before
// the `@` is userinfo, so the string shares our prefix while the real origin is
// attacker.example. Loading it would hand the hermesDesktop preload bridge to an
// attacker-controlled origin.
test('isRendererUrl rejects a userinfo-prefixed lookalike origin', () => {
  const hostile = 'http://127.0.0.1:47891@attacker.example/'

  assert.equal(hostile.startsWith(HTTP_BASE), true, 'precondition: the old prefix check passed this')
  assert.equal(isRendererUrl(hostile, HTTP_BASE), false)
})

test('isRendererUrl rejects other origins, ports, and schemes', () => {
  assert.equal(isRendererUrl('http://attacker.example/', HTTP_BASE), false)
  assert.equal(isRendererUrl('http://127.0.0.1:47892/', HTTP_BASE), false)
  assert.equal(isRendererUrl('https://127.0.0.1:47891/', HTTP_BASE), false)
  assert.equal(isRendererUrl('file:///etc/passwd', HTTP_BASE), false)
})

test('isRendererUrl fails closed on unparseable input', () => {
  assert.equal(isRendererUrl('not a url', HTTP_BASE), false)
  assert.equal(isRendererUrl(`${HTTP_BASE}/`, 'not a url'), false)
})

test('isRendererUrl compares paths for file: bases, whose origin is opaque', () => {
  const index = path.join(path.sep, 'tmp', 'hermes-app', 'dist', 'index.html')
  const other = path.join(path.sep, 'tmp', 'hermes-app', 'dist', 'other.html')
  const base = pathToFileURL(index).toString()

  // Same file, plus the query/hash routing the session windows append.
  assert.equal(isRendererUrl(base, base), true)
  assert.equal(isRendererUrl(`${base}?win=secondary#/abc123`, base), true)

  // A different file under the same directory is NOT the renderer we loaded,
  // and origin equality would have wrongly accepted it: both are "null".
  assert.equal(isRendererUrl(pathToFileURL(other).toString(), base), false)
  assert.equal(isRendererUrl('http://attacker.example/', base), false)
})
