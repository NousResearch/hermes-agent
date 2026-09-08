import assert from 'node:assert/strict'

import { test } from 'vitest'

import { buildHubWindowUrl } from './hub-window'

test('buildHubWindowUrl puts win=skills-hub before the hash (dev server)', () => {
  const url = buildHubWindowUrl({ devServer: 'http://localhost:5174' })

  assert.equal(url, 'http://localhost:5174/?win=skills-hub#/')
})

test('buildHubWindowUrl avoids a double slash when the dev server has a trailing slash', () => {
  const url = buildHubWindowUrl({ devServer: 'http://localhost:5174/' })

  assert.equal(url, 'http://localhost:5174/?win=skills-hub#/')
})

test('buildHubWindowUrl keeps the flag before the hash (HashRouter contract)', () => {
  const url = buildHubWindowUrl({ devServer: 'http://localhost:5174' })

  assert.ok(url.indexOf('?win=skills-hub') < url.indexOf('#'))
})

test('buildHubWindowUrl builds a packaged file URL with the flag before the hash', () => {
  const url = buildHubWindowUrl({ rendererIndexPath: '/opt/app/index.html' })

  assert.match(url, /^file:\/\/.*index\.html\?win=skills-hub#\/$/)
})
