'use strict'

const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const ELECTRON_DIR = __dirname

function readMainSource() {
  return fs.readFileSync(path.join(ELECTRON_DIR, 'main.cjs'), 'utf8').replace(/\r\n/g, '\n')
}

test('pet overlay close does not clear persisted pop-out state during app quit', () => {
  const source = readMainSource()

  assert.match(source, /let isAppQuitting = false/)

  const petOverlayIndex = source.indexOf('// The pet overlay: a single transparent')
  assert.notEqual(petOverlayIndex, -1, 'missing pet overlay section')

  const overlayClosedIndex = source.indexOf("win.on('closed', () => {", petOverlayIndex)
  assert.notEqual(overlayClosedIndex, -1, 'missing pet overlay closed handler')
  const overlayClosedSnippet = source.slice(overlayClosedIndex, overlayClosedIndex + 700)

  assert.match(
    overlayClosedSnippet,
    /mainWindow && !mainWindow\.isDestroyed\(\) && !isAppQuitting/,
    'overlay closed handler must not send pop-in while the app is quitting'
  )
  assert.match(overlayClosedSnippet, /type: 'pop-in'/)

  const beforeQuitIndex = source.indexOf("app.on('before-quit'")
  assert.notEqual(beforeQuitIndex, -1, 'missing before-quit handler')
  const beforeQuitSnippet = source.slice(beforeQuitIndex, beforeQuitIndex + 350)
  const flagIndex = beforeQuitSnippet.indexOf('isAppQuitting = true')
  const closeIndex = beforeQuitSnippet.indexOf('closePetOverlay()')

  assert.notEqual(flagIndex, -1, 'before-quit must mark the app as quitting')
  assert.notEqual(closeIndex, -1, 'before-quit must close the pet overlay')
  assert.ok(flagIndex < closeIndex, 'quit flag must be set before closing the overlay')
})

test('non-mac primary-window close is treated as app quit for pet overlay restore', () => {
  const source = readMainSource()
  const closeIndex = source.indexOf("mainWindow.on('close', () => {")

  assert.notEqual(closeIndex, -1, 'missing main window close handler')
  const closeSnippet = source.slice(closeIndex, closeIndex + 350)

  assert.match(closeSnippet, /schedulePersistWindowState\.flush\(\)/)
  assert.match(closeSnippet, /if \(!IS_MAC\) \{\s*isAppQuitting = true\s*\}/)
})
