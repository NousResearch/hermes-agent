/**
 * Unit tests for the pure zoom helpers: clamping garbage input, the
 * percent <-> zoom-level conversion the settings UI relies on, and the
 * roundtrip stability of the preset percentages.
 */

import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import test from 'node:test'
import { fileURLToPath } from 'node:url'

import {
  clampZoomLevel,
  installZoomReassertOnWindowEvents,
  percentToZoomLevel,
  ZOOM_REASSERT_WINDOW_EVENTS,
  ZOOM_STORAGE_KEY,
  zoomLevelToPercent
} from './zoom.ts'

test('storage key stays stable so persisted zoom survives upgrades', () => {
  assert.equal(ZOOM_STORAGE_KEY, 'hermes:desktop:zoomLevel')
})

test('clampZoomLevel rejects garbage and enforces bounds', () => {
  assert.equal(clampZoomLevel(NaN), 0)
  assert.equal(clampZoomLevel(Infinity), 0)
  assert.equal(clampZoomLevel(undefined), 0)
  assert.equal(clampZoomLevel('2'), 0)
  assert.equal(clampZoomLevel(0.3), 0.3)
  assert.equal(clampZoomLevel(-42), -9)
  assert.equal(clampZoomLevel(42), 9)
})

test('level 0 is exactly 100 percent', () => {
  assert.equal(zoomLevelToPercent(0), 100)
  assert.equal(percentToZoomLevel(100), 0)
})

test('percentToZoomLevel rejects garbage', () => {
  assert.equal(percentToZoomLevel(NaN), 0)
  assert.equal(percentToZoomLevel(0), 0)
  assert.equal(percentToZoomLevel(-50), 0)
  assert.equal(percentToZoomLevel(undefined), 0)
})

test('preset percentages roundtrip within rounding', () => {
  for (const percent of [90, 100, 110, 125, 150, 175]) {
    assert.equal(zoomLevelToPercent(percentToZoomLevel(percent)), percent)
  }
})

test('conversion is monotonic across the preset range', () => {
  const levels = [90, 100, 110, 125, 150, 175].map(percentToZoomLevel)

  for (let i = 1; i < levels.length; i++) {
    assert.ok(levels[i] > levels[i - 1])
  }
})

test('extreme percentages clamp to the level bounds', () => {
  assert.equal(percentToZoomLevel(1), -9)
  assert.equal(percentToZoomLevel(1_000_000), 9)
})

test('installZoomReassertOnWindowEvents wires show and restore', () => {
  const handlers = new Map()
  const win = {
    isDestroyed: () => false,
    on(event, listener) {
      handlers.set(event, listener)
    }
  }
  let calls = 0
  installZoomReassertOnWindowEvents(win, () => {
    calls += 1
  })

  assert.deepEqual([...handlers.keys()], [...ZOOM_REASSERT_WINDOW_EVENTS])
  handlers.get('show')()
  handlers.get('restore')()
  assert.equal(calls, 2)
})

test('installZoomReassertOnWindowEvents skips destroyed windows', () => {
  const handlers = new Map()
  let destroyed = false
  const win = {
    isDestroyed: () => destroyed,
    on(event, listener) {
      handlers.set(event, listener)
    }
  }
  let calls = 0
  installZoomReassertOnWindowEvents(win, () => {
    calls += 1
  })
  destroyed = true
  handlers.get('show')()
  assert.equal(calls, 0)
})

// Source assertion (see windows-child-process.test.ts for the established
// pattern): wireCommonWindowHandlers lives in the electron main entry with heavy
// Electron deps, so we assert the wiring contract against source rather than
// booting a BrowserWindow. Locks in that the pet overlay opts OUT of global UI
// zoom while chat windows keep it — the whole reason this fix is scoped.
test('pet overlay opts out of global UI zoom; chat windows keep it', () => {
  const electronDir = path.dirname(fileURLToPath(import.meta.url))
  const source = fs.readFileSync(path.join(electronDir, 'main.ts'), 'utf8').replace(/\r\n/g, '\n')

  // The shared helper gates all zoom wiring behind an opt-out flag.
  assert.match(source, /function wireCommonWindowHandlers\(win, \{ zoom = true \}/)

  // The pet overlay window is the only caller that disables zoom.
  assert.match(source, /wireCommonWindowHandlers\(win, \{ zoom: false \}\)/)

  // Zoom restore now flows through the shared helper, so createWindow must not
  // reassert it directly (that would double-fire and drift from session windows).
  const finishLoad = source.indexOf("mainWindow.webContents.once('did-finish-load'")
  assert.notEqual(finishLoad, -1, 'missing mainWindow did-finish-load handler')
  const snippet = source.slice(finishLoad, finishLoad + 300)
  assert.doesNotMatch(snippet, /restorePersistedZoomLevel\(mainWindow\)/)
})
