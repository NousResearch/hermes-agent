/**
 * Unit tests for the companion-overlay geometry helpers. These cover the logic that
 * protects the popped-out companion: a remembered spot on a since-unplugged monitor
 * must never strand the overlay off-screen (the "companion vanished" bug), and a
 * live re-home must leave an on-screen companion alone.
 */

import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  clampRectToWorkArea,
  companionOverlayClickThrough,
  resolveCompanionOverlayBounds,
  shouldPopInOnOverlayClosed
} from './companion-overlay'

test('companionOverlayClickThrough is off on Linux, where forward:true never re-arms the sprite', () => {
  assert.equal(companionOverlayClickThrough('darwin'), true)
  assert.equal(companionOverlayClickThrough('win32'), true)
  assert.equal(companionOverlayClickThrough('linux'), false)
})

// ─── quit-path pop-in suppression (#55920) ──────────────────────────────────
//
// The overlay's 'closed' handler echoes pop-in so a companion whose window went away
// on its own (⌘W) comes back inside the app. During a quit that echo is a bug:
// popInCompanion() persists $companionOverlayActive=false, so restoreCompanionOverlay() on the
// next boot skips restoration and the companion silently reverts to in-window.

test('the overlay close echo pops the companion back in while the app is alive', () => {
  assert.equal(shouldPopInOnOverlayClosed({ appQuitting: false, mainWindowAlive: true }), true)
})

test('a quit suppresses the pop-in echo, so the popped-out state survives for the next boot', () => {
  // before-quit teardown and the non-macOS primary-window close both latch the
  // quit flag before closeCompanionOverlay() fires the overlay's 'closed' handler.
  assert.equal(shouldPopInOnOverlayClosed({ appQuitting: true, mainWindowAlive: true }), false)
})

test('no live main window means no echo, quit or not', () => {
  assert.equal(shouldPopInOnOverlayClosed({ appQuitting: false, mainWindowAlive: false }), false)
  assert.equal(shouldPopInOnOverlayClosed({ appQuitting: true, mainWindowAlive: false }), false)
})

// A laptop panel left behind after a bigger external monitor is unplugged.
const LAPTOP = [{ workArea: { x: 0, y: 0, width: 1366, height: 728 } }]

// External monitor to the right of the laptop panel, now disconnected.
const LAPTOP_PLUS_EXTERNAL = [
  { workArea: { x: 0, y: 0, width: 1366, height: 728 } },
  { workArea: { x: 1366, y: 0, width: 1920, height: 1040 } }
]

const ANCHOR_ON_LAPTOP = { x: 100, y: 80, width: 1200, height: 700 }
const ANCHOR_ON_EXTERNAL = { x: 1400, y: 100, width: 1200, height: 700 }

const COMPANION_BOUNDS = { x: 200, y: 150, width: 300, height: 400 }

// ─── sanity / garbage ──────────────────────────────────────────────────────

test('resolveCompanionOverlayBounds returns null for missing or garbage input', () => {
  assert.equal(resolveCompanionOverlayBounds(null, LAPTOP, ANCHOR_ON_LAPTOP), null)
  assert.equal(resolveCompanionOverlayBounds(undefined, LAPTOP, ANCHOR_ON_LAPTOP), null)
  assert.equal(resolveCompanionOverlayBounds({ x: NaN, y: 0, width: 100, height: 100 }, LAPTOP, ANCHOR_ON_LAPTOP), null)
  assert.equal(resolveCompanionOverlayBounds({ x: 0, y: 0, width: 'wide', height: 100 }, LAPTOP, ANCHOR_ON_LAPTOP), null)
})

test('resolveCompanionOverlayBounds returns requested unchanged with no displays to validate against', () => {
  assert.equal(resolveCompanionOverlayBounds(COMPANION_BOUNDS, null, ANCHOR_ON_LAPTOP), COMPANION_BOUNDS)
  assert.equal(resolveCompanionOverlayBounds(COMPANION_BOUNDS, [], ANCHOR_ON_LAPTOP), COMPANION_BOUNDS)
})

// ─── on-screen trust ───────────────────────────────────────────────────────

test('an on-screen saved spot is used as-is', () => {
  assert.deepEqual(resolveCompanionOverlayBounds(COMPANION_BOUNDS, LAPTOP, ANCHOR_ON_LAPTOP), COMPANION_BOUNDS)
})

test('a spot on any connected display is used as-is, even if not the anchor display', () => {
  const onExternal = { x: 1400, y: 200, width: 300, height: 400 }

  assert.deepEqual(resolveCompanionOverlayBounds(onExternal, LAPTOP_PLUS_EXTERNAL, ANCHOR_ON_LAPTOP), onExternal)
})

// ─── partly off-screen: clamp the whole rect in (#85092) ─────────────────────

const within = (r, a) => r.x >= a.x && r.y >= a.y && r.x + r.width <= a.x + a.width && r.y + r.height <= a.y + a.height

test('a first pop-out that is mostly off-screen is pulled wholly onto the display', () => {
  // The #85092 repro: 1707x1067 logical work area (2560x1600 @ 150%), overlay
  // spawned from a companion near the bottom-right corner with only a 49px sliver
  // visible. That passes a 48px-overlap rule, so it must be clamped, not trusted.
  const area = { x: 0, y: 0, width: 1707, height: 1067 }
  const resolved = resolveCompanionOverlayBounds({ x: 1658, y: 806, width: 292, height: 408 }, [{ workArea: area }], null)

  assert.deepEqual(resolved, { x: 1707 - 292, y: 1067 - 408, width: 292, height: 408 })
})

test('a sliver of overlap on any side is clamped to the nearest edge, not re-centered', () => {
  for (const sliver of [
    { x: -270, y: 150, width: 300, height: 400 },
    { x: 200, y: -390, width: 300, height: 400 },
    { x: 1360, y: 700, width: 300, height: 400 }
  ]) {
    const resolved = resolveCompanionOverlayBounds(sliver, LAPTOP, ANCHOR_ON_LAPTOP)

    assert.ok(within(resolved, LAPTOP[0].workArea), JSON.stringify(resolved))
    assert.equal(resolved.width, 300)
    assert.equal(resolved.height, 400)
  }
})

test('a rect straddling two displays is clamped into the one it overlaps most', () => {
  // 250px on the laptop, 50px on the external → stays on the laptop.
  const straddle = { x: 1116, y: 200, width: 300, height: 400 }
  const resolved = resolveCompanionOverlayBounds(straddle, LAPTOP_PLUS_EXTERNAL, ANCHOR_ON_EXTERNAL)

  assert.deepEqual(resolved, { x: 1366 - 300, y: 200, width: 300, height: 400 })
})

test('clampRectToWorkArea keeps any rect wholly inside the work area', () => {
  const area = { x: -1920, y: 25, width: 1920, height: 1055 }

  for (const rect of [
    { x: -5000, y: -5000, width: 300, height: 400 },
    { x: 5000, y: 5000, width: 300, height: 400 },
    { x: -1000, y: 500, width: 300, height: 400 },
    { x: -1920, y: 25, width: 4000, height: 3000 },
    { x: -100.6, y: 900.4, width: 299.5, height: 400.2 }
  ]) {
    const clamped = clampRectToWorkArea(rect, area)

    assert.ok(within(clamped, area), JSON.stringify(clamped))
    assert.ok([clamped.x, clamped.y, clamped.width, clamped.height].every(Number.isInteger))
  }

  // Already inside: unchanged.
  assert.deepEqual(clampRectToWorkArea({ x: -1000, y: 500, width: 300, height: 400 }, area), {
    x: -1000,
    y: 500,
    width: 300,
    height: 400
  })
})

// ─── off-screen re-home ────────────────────────────────────────────────────

test('a spot remembered on a since-unplugged external monitor is re-centered on the anchor display', () => {
  // Saved while the external was connected (x 1366+); only the laptop remains.
  const stale = { x: 1500, y: 400, width: 300, height: 400 }

  const resolved = resolveCompanionOverlayBounds(stale, LAPTOP, ANCHOR_ON_LAPTOP)

  assert.deepEqual(resolved, {
    x: Math.round((1366 - 300) / 2),
    y: Math.round((728 - 400) / 2),
    width: 300,
    height: 400
  })
})

test('re-home centers on the display holding the anchor (main window), not the primary', () => {
  const stale = { x: -800, y: 400, width: 300, height: 400 }

  const resolved = resolveCompanionOverlayBounds(stale, LAPTOP_PLUS_EXTERNAL, ANCHOR_ON_EXTERNAL)

  assert.deepEqual(resolved, {
    x: Math.round(1366 + (1920 - 300) / 2),
    y: Math.round((1040 - 400) / 2),
    width: 300,
    height: 400
  })
})

test('re-home falls back to the primary display when the anchor is missing', () => {
  const stale = { x: 1500, y: 400, width: 300, height: 400 }

  const resolved = resolveCompanionOverlayBounds(stale, LAPTOP, null)

  assert.deepEqual(resolved, {
    x: Math.round((1366 - 300) / 2),
    y: Math.round((728 - 400) / 2),
    width: 300,
    height: 400
  })
})

test('re-homed size is capped to the target work area', () => {
  const huge = { x: 1500, y: 400, width: 2000, height: 1500 }

  const resolved = resolveCompanionOverlayBounds(huge, LAPTOP, ANCHOR_ON_LAPTOP)

  assert.deepEqual(resolved, {
    x: 0,
    y: 0,
    width: 1366,
    height: 728
  })
})

// ─── live re-home (display unplugged while the companion is popped out) ──────────

test('a still-on-screen companion is left exactly where it is', () => {
  assert.deepEqual(resolveCompanionOverlayBounds(COMPANION_BOUNDS, LAPTOP, ANCHOR_ON_LAPTOP), COMPANION_BOUNDS)
})

test('a companion stranded by an unplugged display is pulled back onto the anchor display', () => {
  const stranded = { x: 1500, y: 400, width: 300, height: 400 }

  const resolved = resolveCompanionOverlayBounds(stranded, LAPTOP, ANCHOR_ON_LAPTOP)

  assert.deepEqual(resolved, {
    x: Math.round((1366 - 300) / 2),
    y: Math.round((728 - 400) / 2),
    width: 300,
    height: 400
  })
})
