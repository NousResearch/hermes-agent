import assert from 'node:assert/strict'

import { test } from 'vitest'

import { buildCanvasWindowUrl, isCanvasWindowTab } from './canvas-windows'

const tab = { provider: 'pen', docId: 'doc-1', title: 'Landing page', url: 'https://app.pen.dev/new?embed' }

test('buildCanvasWindowUrl puts win=canvas and the tab before the hash (dev server)', () => {
  const url = buildCanvasWindowUrl(tab, { devServer: 'http://localhost:5173/' })
  const parsed = new URL(url)

  assert.equal(parsed.origin + parsed.pathname, 'http://localhost:5173/')
  assert.equal(parsed.hash, '#/')
  assert.ok(url.indexOf('?win=canvas') < url.indexOf('#'))
  assert.equal(parsed.searchParams.get('win'), 'canvas')
  assert.equal(parsed.searchParams.get('provider'), 'pen')
  assert.equal(parsed.searchParams.get('doc'), 'doc-1')
  assert.equal(parsed.searchParams.get('title'), 'Landing page')
  // The editor URL's own query survives a round trip through ours.
  assert.equal(parsed.searchParams.get('url'), 'https://app.pen.dev/new?embed')
})

test('buildCanvasWindowUrl builds a packaged file URL with the flag before the hash', () => {
  const url = buildCanvasWindowUrl(tab, { rendererIndexPath: '/opt/app/index.html' })

  assert.match(url, /^file:\/\/.*index\.html\?win=canvas&provider=pen&.*#\/$/)
})

test('isCanvasWindowTab accepts a docked tab and rejects a provider-less one', () => {
  assert.equal(isCanvasWindowTab(tab), true)
  assert.equal(isCanvasWindowTab({ ...tab, provider: ' ' }), false)
  assert.equal(isCanvasWindowTab({ ...tab, url: 7 }), false)
  assert.equal(isCanvasWindowTab(null), false)
})
