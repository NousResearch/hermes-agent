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
  isDebouncedReassertEvent,
  percentToZoomLevel,
  ZOOM_REASSERT_MAX_SETTLE_CHECKS,
  ZOOM_REASSERT_SETTLE_DELAY_MS,
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

test('focus event reasserts zoom immediately without debounce on Windows (high-DPI alt-tab, #50837)', () => {
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

  // focus on Windows triggers immediate reassert — no timer involved
  handlers.get('focus')()
  assert.equal(calls, 1)
})

test('isDebouncedReassertEvent debounces focus only on Linux, not Windows/macOS', () => {
  assert.equal(isDebouncedReassertEvent('focus', 'linux'), true)
  assert.equal(isDebouncedReassertEvent('focus', 'win32'), false)
  assert.equal(isDebouncedReassertEvent('focus', 'darwin'), false)
  assert.equal(isDebouncedReassertEvent('resize', 'linux'), true)
  assert.equal(isDebouncedReassertEvent('resize', 'win32'), true)
  assert.equal(isDebouncedReassertEvent('show', 'linux'), false)
  assert.equal(isDebouncedReassertEvent('restore', 'linux'), false)
})

test('installZoomReassertOnWindowEvents debounces Linux resize, move, and focus events at the trailing edge', () => {
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

    // focus on Linux is also debounced — a session switch fires focus but
    // the reassert must not jump the zoom mid-interaction.
    handlers.get('focus')()
    vi.advanceTimersByTime(ZOOM_RESIZE_REASSERT_DELAY_MS / 2)
    assert.equal(calls, 1)
    vi.advanceTimersByTime(ZOOM_RESIZE_REASSERT_DELAY_MS / 2)
    assert.equal(calls, 2)

    handlers.get('resize')()
    destroyed = true
    vi.advanceTimersByTime(ZOOM_RESIZE_REASSERT_DELAY_MS)
    assert.equal(calls, 2)
  } finally {
    vi.useRealTimers()
  }
})

test('installZoomReassertOnWindowEvents re-verifies Linux zoom after the debounced re-assert so a dropped re-apply is retried', () => {
  vi.useFakeTimers()

  try {
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
      'linux'
    )

    handlers.get('resize')()
    vi.advanceTimersByTime(ZOOM_RESIZE_REASSERT_DELAY_MS)
    assert.equal(calls, 1, 'debounced re-assert fires once')

    // Cosmic tiling can drop the just-applied zoom mid-reconfigure; the
    // settle-verify chain re-asserts on a bounded schedule until it converges:
    // one initial re-assert plus ZOOM_REASSERT_MAX_SETTLE_CHECKS follow-ups.
    for (let i = 1; i <= ZOOM_REASSERT_MAX_SETTLE_CHECKS; i++) {
      vi.advanceTimersByTime(ZOOM_REASSERT_SETTLE_DELAY_MS)
      assert.equal(calls, 1 + i, `settle check ${i} re-asserts`)
    }

    vi.advanceTimersByTime(ZOOM_REASSERT_SETTLE_DELAY_MS * 2)
    assert.equal(calls, 1 + ZOOM_REASSERT_MAX_SETTLE_CHECKS, 'settle chain is bounded and stops')
  } finally {
    vi.useRealTimers()
  }
})

test('installZoomReassertOnWindowEvents resets the settle chain on a fresh re-assert', () => {
  vi.useFakeTimers()

  try {
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
      'linux'
    )

    handlers.get('show')()
    vi.advanceTimersByTime(ZOOM_REASSERT_SETTLE_DELAY_MS)
    assert.equal(calls, 2, 'initial re-assert plus one settle check')

    // A second transition while the first settle chain is still pending must
    // restart the chain from zero instead of exhausting the budget.
    handlers.get('restore')()
    assert.equal(calls, 3, 'fresh transition re-asserts immediately')

    for (let i = 1; i <= ZOOM_REASSERT_MAX_SETTLE_CHECKS; i++) {
      vi.advanceTimersByTime(ZOOM_REASSERT_SETTLE_DELAY_MS)
      assert.equal(calls, 3 + i, `restarted chain settle check ${i} re-asserts`)
    }

    vi.advanceTimersByTime(ZOOM_REASSERT_SETTLE_DELAY_MS * 2)
    assert.equal(calls, 3 + ZOOM_REASSERT_MAX_SETTLE_CHECKS, 'restarted chain is bounded too')
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
