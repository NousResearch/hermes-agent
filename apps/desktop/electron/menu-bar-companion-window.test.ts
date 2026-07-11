import assert from 'node:assert/strict'

import { test } from 'vitest'

import { buildMenuBarCompanionWindowUrl, positionMenuBarCompanionWindowBounds } from './menu-bar-companion-window'

test('buildMenuBarCompanionWindowUrl uses dev server route', () => {
  assert.equal(
    buildMenuBarCompanionWindowUrl({ devServer: 'http://127.0.0.1:5174' }),
    'http://127.0.0.1:5174/?win=menu-bar-companion#/'
  )
})

test('buildMenuBarCompanionWindowUrl uses file URL in packaged mode', () => {
  const url = buildMenuBarCompanionWindowUrl({
    rendererIndexPath: '/tmp/dist/index.html'
  })

  assert.match(url, /^file:\/\/\/tmp\/dist\/index\.html\?win=menu-bar-companion#\/$/)
})

test('positionMenuBarCompanionWindowBounds prefers below tray when it fits', () => {
  const bounds = positionMenuBarCompanionWindowBounds({
    displayBounds: { x: 0, y: 0, width: 1440, height: 900 },
    trayBounds: { x: 1200, y: 0, width: 24, height: 24 },
    windowSize: { width: 420, height: 720 }
  })

  assert.equal(bounds.width, 420)
  assert.equal(bounds.height, 720)
  assert.equal(bounds.y, 32)
  assert.ok(bounds.x >= 8)
  assert.ok(bounds.x + bounds.width <= 1440 - 8)
})

test('positionMenuBarCompanionWindowBounds flips above when below would overflow', () => {
  const bounds = positionMenuBarCompanionWindowBounds({
    displayBounds: { x: 0, y: 0, width: 800, height: 600 },
    trayBounds: { x: 400, y: 500, width: 20, height: 20 },
    windowSize: { width: 420, height: 500 }
  })

  assert.ok(bounds.y < 500)
})
