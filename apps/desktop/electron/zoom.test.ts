/**
 * Unit tests for the pure zoom helpers: clamping garbage input, the
 * percent <-> zoom-level conversion the settings UI relies on, and the
 * roundtrip stability of the preset percentages.
 */

import assert from 'node:assert/strict'

import { test, vi } from 'vitest'

import {
  applyZoomLevel,
  clampZoomLevel,
  DEFAULT_ZOOM_LEVEL,
  installZoomReassertOnWindowEvents,
  percentToZoomLevel,
  ZOOM_RESIZE_REASSERT_DELAY_MS,
  ZOOM_STEP,
  ZOOM_STORAGE_KEY,
  zoomLevelToPercent,
  zoomReassertWindowEvents,
  zoomWiringForWindowKind
} from './zoom'

test('storage key stays stable so persisted zoom survives upgrades', () => {
  assert.equal(ZOOM_STORAGE_KEY, 'hermes:desktop:zoomLevel')
})

test('default zoom matches the Appearance 90% preset', () => {
  assert.equal(ZOOM_STEP, 0.1)
  assert.equal(zoomLevelToPercent(DEFAULT_ZOOM_LEVEL), 90)
  assert.equal(DEFAULT_ZOOM_LEVEL, percentToZoomLevel(90))
})

test('clampZoomLevel rejects garbage and enforces bounds', () => {
  assert.equal(clampZoomLevel(NaN), DEFAULT_ZOOM_LEVEL)
  assert.equal(clampZoomLevel(Infinity), DEFAULT_ZOOM_LEVEL)
  assert.equal(clampZoomLevel(undefined), DEFAULT_ZOOM_LEVEL)
  assert.equal(clampZoomLevel('2'), DEFAULT_ZOOM_LEVEL)
  assert.equal(clampZoomLevel(0.3), 0.3)
  assert.equal(clampZoomLevel(-42), -9)
  assert.equal(clampZoomLevel(42), 9)
})

test('level 0 is exactly 100 percent (Chromium actual-size baseline)', () => {
  assert.equal(zoomLevelToPercent(0), 100)
  assert.equal(percentToZoomLevel(100), 0)
})

test('percentToZoomLevel rejects garbage by falling back to the shipped default', () => {
  assert.equal(percentToZoomLevel(NaN), DEFAULT_ZOOM_LEVEL)
  assert.equal(percentToZoomLevel(0), DEFAULT_ZOOM_LEVEL)
  assert.equal(percentToZoomLevel(-50), DEFAULT_ZOOM_LEVEL)
  assert.equal(percentToZoomLevel(undefined), DEFAULT_ZOOM_LEVEL)
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

test('installZoomReassertOnWindowEvents wires show, restore, focus, resize, and cross-display moves on macOS and Windows', () => {
  const handlers = new Map()

  const win = {
    isDestroyed: () => false,
    on(event, listener) {
      handlers.set(event, listener)
    }
  }

  let calls = 0
  installZoomReassertOnWindowEvents(
    win,
    () => {
      calls += 1
    },
    'win32'
  )

  assert.deepEqual([...handlers.keys()], zoomReassertWindowEvents('win32'))
  handlers.get('show')()
  handlers.get('restore')()
  handlers.get('focus')()
  handlers.get('resized')()
  handlers.get('moved')()
  assert.equal(calls, 5)
})

test('focus event reasserts zoom immediately without debounce (Windows high-DPI alt-tab, #50837)', () => {
  const handlers = new Map()

  const win = {
    isDestroyed: () => false,
    on(event, listener) {
      handlers.set(event, listener)
    }
  }

  let calls = 0
  installZoomReassertOnWindowEvents(
    win,
    () => {
      calls += 1
    },
    'win32'
  )

  // focus should trigger immediate reassert — no timer involved
  handlers.get('focus')()
  assert.equal(calls, 1)
})

test('installZoomReassertOnWindowEvents debounces Linux resize and move events at the trailing edge', () => {
  vi.useFakeTimers()

  try {
    const handlers = new Map()
    let destroyed = false

    const win = {
      isDestroyed: () => destroyed,
      on(event, listener) {
        handlers.set(event, listener)
      }
    }

    let calls = 0

    installZoomReassertOnWindowEvents(
      win,
      () => {
        calls += 1
      },
      'linux'
    )

    assert.deepEqual([...handlers.keys()], zoomReassertWindowEvents('linux'))
    handlers.get('resize')()
    vi.advanceTimersByTime(ZOOM_RESIZE_REASSERT_DELAY_MS / 2)
    handlers.get('move')()
    vi.advanceTimersByTime(ZOOM_RESIZE_REASSERT_DELAY_MS / 2)
    assert.equal(calls, 0)
    vi.advanceTimersByTime(ZOOM_RESIZE_REASSERT_DELAY_MS / 2)
    assert.equal(calls, 1)

    handlers.get('resize')()
    destroyed = true
    vi.advanceTimersByTime(ZOOM_RESIZE_REASSERT_DELAY_MS)
    assert.equal(calls, 1)
  } finally {
    vi.useRealTimers()
  }
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

// Zoom-wiring contract: chat windows keep global UI zoom while fixed-size
// helper windows opt out. Tested via the extracted config — no source-text regex.
test('chat windows opt into zoom', () => {
  assert.deepEqual(zoomWiringForWindowKind('chat'), { zoom: true })
})

test('pet overlay opts out of zoom', () => {
  assert.deepEqual(zoomWiringForWindowKind('petOverlay'), { zoom: false })
})

test('wake indicator opts out of zoom', () => {
  assert.deepEqual(zoomWiringForWindowKind('wakeIndicator'), { zoom: false })
})

test('unknown window kinds default to chat (zoom enabled)', () => {
  assert.deepEqual(zoomWiringForWindowKind('unknown'), { zoom: true })
  assert.deepEqual(zoomWiringForWindowKind(undefined), { zoom: true })
})

// The UI Scale settings control drifts out of sync after a restart when zoom
// is applied to the window but the renderer is never told: its $zoomPercent
// store (see store/zoom.ts) only updates from zoom.get() (once, on load) and
// 'hermes:zoom:changed' events. applyZoomLevel is the single funnel every zoom
// path (user set, restore-on-load, lifecycle re-assert) shares, so applying a
// level always notifies — the regression can't come back by forgetting a send.
function fakeWebContents() {
  const calls: Array<[string, ...unknown[]]> = []

  return {
    calls,
    setZoomLevel: (level: number) => calls.push(['setZoomLevel', level]),
    send: (channel: string, payload: unknown) => calls.push(['send', channel, payload])
  }
}

test('applyZoomLevel applies the level then notifies the renderer', () => {
  const wc = fakeWebContents()
  const applied = applyZoomLevel(wc, 3)

  assert.equal(applied, 3)
  assert.deepEqual(wc.calls, [
    ['setZoomLevel', 3],
    ['send', 'hermes:zoom:changed', { level: 3, percent: zoomLevelToPercent(3) }]
  ])
})

test('applyZoomLevel clamps garbage before applying and notifying', () => {
  const wc = fakeWebContents()
  const applied = applyZoomLevel(wc, 999)
  const clamped = clampZoomLevel(999)

  assert.equal(applied, clamped)
  assert.deepEqual(wc.calls, [
    ['setZoomLevel', clamped],
    ['send', 'hermes:zoom:changed', { level: clamped, percent: zoomLevelToPercent(clamped) }]
  ])
})
