/**
 * Tests for electron/window-state.cjs.
 *
 * Run with: node --test electron/window-state.test.cjs
 * (Wired into npm test:desktop:platforms in package.json.)
 *
 * Covers the pure validation logic behind window size/position restore:
 * sanitizing persisted state, off-screen detection across displays, and the
 * resulting BrowserWindow options (including the off-screen fallback).
 */

const test = require('node:test')
const assert = require('node:assert/strict')

const {
  DEFAULT_WIDTH,
  DEFAULT_HEIGHT,
  MIN_WIDTH,
  MIN_HEIGHT,
  sanitizeWindowState,
  boundsVisibleOnDisplays,
  computeWindowOptions
} = require('./window-state.cjs')

// A typical single 1920x1080 monitor (work area trimmed for the taskbar).
const PRIMARY = [{ workArea: { x: 0, y: 0, width: 1920, height: 1040 } }]

// --- sanitizeWindowState ---

test('sanitizeWindowState returns null for missing/garbage input', () => {
  assert.equal(sanitizeWindowState(null), null)
  assert.equal(sanitizeWindowState(undefined), null)
  assert.equal(sanitizeWindowState('nope'), null)
  assert.equal(sanitizeWindowState(42), null)
  assert.equal(sanitizeWindowState({}), null)
  assert.equal(sanitizeWindowState({ width: 'wide', height: 800 }), null)
  assert.equal(sanitizeWindowState({ width: NaN, height: 800 }), null)
  assert.equal(sanitizeWindowState({ width: 1000 }), null) // height missing
})

test('sanitizeWindowState keeps a valid full state', () => {
  assert.deepEqual(sanitizeWindowState({ x: 100, y: 50, width: 1400, height: 900, isMaximized: true }), {
    x: 100,
    y: 50,
    width: 1400,
    height: 900,
    isMaximized: true
  })
})

test('sanitizeWindowState clamps width/height to the minimums', () => {
  const state = sanitizeWindowState({ width: 10, height: 10 })
  assert.equal(state.width, MIN_WIDTH)
  assert.equal(state.height, MIN_HEIGHT)
})

test('sanitizeWindowState rounds fractional bounds (HiDPI scaling)', () => {
  assert.deepEqual(sanitizeWindowState({ x: 100.6, y: 50.2, width: 1400.4, height: 900.7 }), {
    x: 101,
    y: 50,
    width: 1400,
    height: 901,
    isMaximized: false
  })
})

test('sanitizeWindowState drops a partial position but keeps the size', () => {
  const state = sanitizeWindowState({ x: 100, width: 1400, height: 900 })
  assert.equal(state.x, undefined)
  assert.equal(state.y, undefined)
  assert.equal(state.width, 1400)
})

test('sanitizeWindowState defaults isMaximized to false', () => {
  assert.equal(sanitizeWindowState({ width: 1400, height: 900 }).isMaximized, false)
  assert.equal(sanitizeWindowState({ width: 1400, height: 900, isMaximized: 'yes' }).isMaximized, false)
})

// --- boundsVisibleOnDisplays ---

test('boundsVisibleOnDisplays accepts a window fully on the primary display', () => {
  assert.equal(boundsVisibleOnDisplays({ x: 100, y: 100, width: 1220, height: 800 }, PRIMARY), true)
})

test('boundsVisibleOnDisplays rejects a window past the right edge (unplugged monitor)', () => {
  assert.equal(boundsVisibleOnDisplays({ x: 3000, y: 100, width: 1220, height: 800 }, PRIMARY), false)
})

test('boundsVisibleOnDisplays rejects a window above the top edge', () => {
  assert.equal(boundsVisibleOnDisplays({ x: 100, y: -900, width: 1220, height: 800 }, PRIMARY), false)
})

test('boundsVisibleOnDisplays accepts a window on a secondary display', () => {
  const displays = [
    { workArea: { x: 0, y: 0, width: 1920, height: 1040 } },
    { workArea: { x: 1920, y: 0, width: 2560, height: 1400 } }
  ]
  assert.equal(boundsVisibleOnDisplays({ x: 2200, y: 200, width: 1220, height: 800 }, displays), true)
})

test('boundsVisibleOnDisplays rejects a sliver of overlap below the threshold', () => {
  // Only ~10px peeking onto the primary display from the right.
  assert.equal(boundsVisibleOnDisplays({ x: 1910, y: 100, width: 1220, height: 800 }, PRIMARY), false)
})

test('boundsVisibleOnDisplays returns false with no displays or bad bounds', () => {
  assert.equal(boundsVisibleOnDisplays({ x: 0, y: 0, width: 1220, height: 800 }, []), false)
  assert.equal(boundsVisibleOnDisplays(null, PRIMARY), false)
  assert.equal(boundsVisibleOnDisplays({ x: 'a', y: 0, width: 1220, height: 800 }, PRIMARY), false)
})

// --- computeWindowOptions ---

test('computeWindowOptions falls back to defaults when there is no saved state', () => {
  assert.deepEqual(computeWindowOptions(null, PRIMARY), {
    width: DEFAULT_WIDTH,
    height: DEFAULT_HEIGHT
  })
})

test('computeWindowOptions restores an on-screen saved position', () => {
  const saved = sanitizeWindowState({ x: 200, y: 150, width: 1400, height: 900 })
  assert.deepEqual(computeWindowOptions(saved, PRIMARY), {
    width: 1400,
    height: 900,
    x: 200,
    y: 150
  })
})

test('computeWindowOptions keeps the size but drops an off-screen position', () => {
  const saved = sanitizeWindowState({ x: 5000, y: 150, width: 1400, height: 900 })
  const opts = computeWindowOptions(saved, PRIMARY)
  assert.equal(opts.width, 1400)
  assert.equal(opts.height, 900)
  assert.equal(opts.x, undefined) // off-screen → Electron centers instead
  assert.equal(opts.y, undefined)
})

test('computeWindowOptions restores size when only a size was saved (no position)', () => {
  const saved = sanitizeWindowState({ width: 1400, height: 900 })
  assert.deepEqual(computeWindowOptions(saved, PRIMARY), { width: 1400, height: 900 })
})
