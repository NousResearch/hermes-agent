const assert = require('node:assert/strict')
const test = require('node:test')

const { ZOOM_LEVEL_MIN, ZOOM_LEVEL_MAX, clampZoomLevel, createZoomController } = require('./zoom.cjs')

// A minimal fake window whose zoom can be mutated out from under the controller
// (mirroring Chromium resetting zoom on a hash-route navigation), and which
// records every setZoom it receives.
function makeFakeWindow(initialZoom = 0) {
  let zoom = initialZoom
  const sets = []

  return {
    getZoom: () => zoom,
    setZoom(level) {
      zoom = level
      sets.push(level)
    },
    // Test helper: simulate Chromium resetting this window's zoom on navigation.
    resetZoom(level = 0) {
      zoom = level
    },
    sets
  }
}

// Build a controller wired to an in-memory persistence cell and a fixed window
// list, returning the controller plus the test handles.
function makeController({ persisted = null, windows = [] } = {}) {
  let stored = persisted
  const writes = []

  const controller = createZoomController({
    readLevel: () => stored,
    writeLevel: level => {
      stored = level
      writes.push(level)
    },
    getWindows: () => windows,
    getZoom: win => win.getZoom(),
    setZoom: (win, level) => win.setZoom(level)
  })

  return { controller, writes, getStored: () => stored }
}

test('clampZoomLevel coerces non-finite input to 0', () => {
  assert.equal(clampZoomLevel(NaN), 0)
  assert.equal(clampZoomLevel(undefined), 0)
  assert.equal(clampZoomLevel('nope'), 0)
})

test('clampZoomLevel keeps the level within Chromium bounds', () => {
  assert.equal(clampZoomLevel(2), 2)
  assert.equal(clampZoomLevel(ZOOM_LEVEL_MAX + 5), ZOOM_LEVEL_MAX)
  assert.equal(clampZoomLevel(ZOOM_LEVEL_MIN - 5), ZOOM_LEVEL_MIN)
})

test('controller seeds its level from persistence at construction', () => {
  const { controller } = makeController({ persisted: 2 })

  assert.equal(controller.getLevel(), 2)
})

test('controller clamps a garbage persisted value to 0', () => {
  const { controller } = makeController({ persisted: 'corrupt' })

  assert.equal(controller.getLevel(), 0)
})

test('set persists the new level and applies it to every window', () => {
  const a = makeFakeWindow(0)
  const b = makeFakeWindow(0)
  const { controller, writes } = makeController({ windows: [a, b] })

  controller.set(2)

  assert.equal(controller.getLevel(), 2)
  assert.deepEqual(writes, [2], 'persisted exactly once')
  assert.equal(a.getZoom(), 2)
  assert.equal(b.getZoom(), 2)
})

test('set clamps before persisting and applying', () => {
  const win = makeFakeWindow(0)
  const { controller, getStored } = makeController({ windows: [win] })

  controller.set(99)

  assert.equal(controller.getLevel(), ZOOM_LEVEL_MAX)
  assert.equal(getStored(), ZOOM_LEVEL_MAX)
  assert.equal(win.getZoom(), ZOOM_LEVEL_MAX)
})

test('set does not re-persist when the clamped level is unchanged', () => {
  const win = makeFakeWindow(0)
  const { controller, writes } = makeController({ persisted: 2, windows: [win] })

  controller.set(2)

  assert.deepEqual(writes, [], 'no write when the level is the same')
  assert.equal(win.getZoom(), 2, 'but the window is still brought into line')
})

test('apply re-asserts the level after a route reset zeroed the window', () => {
  // Regression for issue #40166: Chromium tracks zoom per hash route and resets
  // it on navigation, so the controller re-applies on did-navigate-in-page.
  const win = makeFakeWindow(0)
  const { controller } = makeController({ persisted: 2, windows: [win] })

  controller.apply(win) // initial load brings the window to 2
  assert.equal(win.getZoom(), 2)

  win.resetZoom(0) // simulate a hash-route navigation resetting zoom
  controller.apply(win)

  assert.equal(win.getZoom(), 2, 'zoom restored after the route change')
})

test('apply is a no-op (no setZoom) when the window already matches', () => {
  const win = makeFakeWindow(2)
  const { controller } = makeController({ persisted: 2, windows: [win] })

  controller.apply(win)

  assert.deepEqual(win.sets, [], 'no redundant setZoom that would fire a zoom event')
})

test('a window opened after a zoom change adopts the current level on apply', () => {
  const existing = makeFakeWindow(0)
  const windows = [existing]
  const { controller } = makeController({ windows })

  controller.set(3)
  assert.equal(existing.getZoom(), 3)

  // A second window opens later and is wired via apply() on did-finish-load.
  const opened = makeFakeWindow(0)
  windows.push(opened)
  controller.apply(opened)

  assert.equal(opened.getZoom(), 3, 'new window picks up the app-level zoom')
})
