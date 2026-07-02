/**
 * Unit tests for the pure window-state geometry helpers. These cover the logic
 * that protects the user: garbage rejection, off-screen fallback, oversized
 * clamping, and the debounce that collapses mid-drag write storms.
 */

const test = require('node:test')
const assert = require('node:assert/strict')

const {
  DEFAULT_WIDTH,
  DEFAULT_HEIGHT,
  MIN_WIDTH,
  MIN_HEIGHT,
  sanitizeWindowState,
  onScreen,
  computeWindowOptions,
  debounce
} = require('./window-state.cjs')

// A single 1920×1080 monitor (work area trimmed for the taskbar).
const PRIMARY = [{ workArea: { x: 0, y: 0, width: 1920, height: 1040 } }]
// A laptop panel left behind after a bigger external monitor is unplugged.
const LAPTOP = [{ workArea: { x: 0, y: 0, width: 1366, height: 728 } }]

// ─── sanitizeWindowState ───────────────────────────────────────────────────

test('sanitizeWindowState rejects missing/garbage input', () => {
  for (const bad of [
    null,
    undefined,
    'nope',
    42,
    {},
    { width: 'x', height: 800 },
    { width: NaN, height: 800 },
    { width: 1000 }
  ]) {
    assert.equal(sanitizeWindowState(bad), null)
  }
})

test('sanitizeWindowState keeps a valid full state and rounds HiDPI fractions', () => {
  assert.deepEqual(sanitizeWindowState({ x: 100.6, y: 50.2, width: 1400.4, height: 900.7, isMaximized: true }), {
    x: 101,
    y: 50,
    width: 1400,
    height: 901,
    isMaximized: true
  })
})

test('sanitizeWindowState floors size to the minimums', () => {
  const state = sanitizeWindowState({ width: 10, height: 10 })
  assert.equal(state.width, MIN_WIDTH)
  assert.equal(state.height, MIN_HEIGHT)
})

test('sanitizeWindowState drops a partial position but keeps the size', () => {
  assert.deepEqual(sanitizeWindowState({ x: 100, width: 1400, height: 900 }), {
    width: 1400,
    height: 900,
    isMaximized: false
  })
})

test('sanitizeWindowState treats isMaximized strictly', () => {
  assert.equal(sanitizeWindowState({ width: 1400, height: 900, isMaximized: 'yes' }).isMaximized, false)
})

// ─── onScreen ──────────────────────────────────────────────────────────────

test('onScreen accepts a window on the primary or a secondary display', () => {
  const dual = [...PRIMARY, { workArea: { x: 1920, y: 0, width: 2560, height: 1400 } }]
  assert.equal(onScreen({ x: 100, y: 100, width: 1220, height: 800 }, PRIMARY), true)
  assert.equal(onScreen({ x: 2200, y: 200, width: 1220, height: 800 }, dual), true)
})

test('onScreen rejects off-screen, slivers, and bad input', () => {
  assert.equal(onScreen({ x: 3000, y: 100, width: 1220, height: 800 }, PRIMARY), false) // past right edge
  assert.equal(onScreen({ x: 100, y: -900, width: 1220, height: 800 }, PRIMARY), false) // above top
  assert.equal(onScreen({ x: 1910, y: 100, width: 1220, height: 800 }, PRIMARY), false) // ~10px sliver
  assert.equal(onScreen({ x: 0, y: 0, width: 1220, height: 800 }, []), false)
  assert.equal(onScreen({ x: 0, y: 0, width: 1220, height: 800 }, null), false)
})

// ─── computeWindowOptions ──────────────────────────────────────────────────

test('computeWindowOptions falls back to defaults with no saved state', () => {
  assert.deepEqual(computeWindowOptions(null, PRIMARY), { width: DEFAULT_WIDTH, height: DEFAULT_HEIGHT })
})

test('computeWindowOptions restores an on-screen position', () => {
  const saved = sanitizeWindowState({ x: 200, y: 150, width: 1400, height: 900 })
  assert.deepEqual(computeWindowOptions(saved, PRIMARY), { width: 1400, height: 900, x: 200, y: 150 })
})

test('computeWindowOptions keeps the size but drops an off-screen position', () => {
  const saved = sanitizeWindowState({ x: 5000, y: 150, width: 1400, height: 900 })
  assert.deepEqual(computeWindowOptions(saved, PRIMARY), { width: 1400, height: 900 })
})

test('computeWindowOptions clamps a size larger than the only display', () => {
  const saved = sanitizeWindowState({ width: 2560, height: 1440 })
  assert.deepEqual(computeWindowOptions(saved, LAPTOP), { width: 1366, height: 728 })
})

test('computeWindowOptions keeps the MIN floor on a sub-minimum display', () => {
  const tiny = [{ workArea: { x: 0, y: 0, width: 360, height: 480 } }]
  const saved = sanitizeWindowState({ width: 2000, height: 1500 })
  assert.deepEqual(computeWindowOptions(saved, tiny), { width: MIN_WIDTH, height: MIN_HEIGHT })
})

test('computeWindowOptions does not clamp when displays are unknown', () => {
  const saved = sanitizeWindowState({ width: 2560, height: 1440 })
  assert.deepEqual(computeWindowOptions(saved, []), { width: 2560, height: 1440 })
})

// ─── debounce ──────────────────────────────────────────────────────────────

test('debounce coalesces a burst into one trailing run', t => {
  t.mock.timers.enable({ apis: ['setTimeout'] })
  let calls = 0
  const d = debounce(() => {
    calls += 1
  }, 250)

  d()
  d()
  d()
  assert.equal(calls, 0)
  t.mock.timers.tick(249)
  assert.equal(calls, 0)
  t.mock.timers.tick(1)
  assert.equal(calls, 1)
})

test('debounce.flush runs now and cancels the pending timer', t => {
  t.mock.timers.enable({ apis: ['setTimeout'] })
  let calls = 0
  const d = debounce(() => {
    calls += 1
  }, 250)

  d()
  d.flush()
  assert.equal(calls, 1)
  t.mock.timers.tick(1000)
  assert.equal(calls, 1)
})
