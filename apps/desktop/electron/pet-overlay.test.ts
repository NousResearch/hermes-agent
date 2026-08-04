/**
 * Unit tests for the pet-overlay geometry helpers. These cover the logic that
 * protects the popped-out pet: a remembered spot on a since-unplugged monitor
 * must never strand the overlay off-screen (the "pet vanished" bug), and a
 * live re-home must leave an on-screen pet alone.
 */

import assert from 'node:assert/strict'

import { test } from 'vitest'

import { resolvePetOverlayBounds } from './pet-overlay'

// A laptop panel left behind after a bigger external monitor is unplugged.
const LAPTOP = [{ workArea: { x: 0, y: 0, width: 1366, height: 728 } }]
// External monitor to the right of the laptop panel, now disconnected.
const LAPTOP_PLUS_EXTERNAL = [
  { workArea: { x: 0, y: 0, width: 1366, height: 728 } },
  { workArea: { x: 1366, y: 0, width: 1920, height: 1040 } }
]

const ANCHOR_ON_LAPTOP = { x: 100, y: 80, width: 1200, height: 700 }
const ANCHOR_ON_EXTERNAL = { x: 1400, y: 100, width: 1200, height: 700 }

const PET_BOUNDS = { x: 200, y: 150, width: 300, height: 400 }

// ─── sanity / garbage ──────────────────────────────────────────────────────

test('resolvePetOverlayBounds returns null for missing or garbage input', () => {
  assert.equal(resolvePetOverlayBounds(null, LAPTOP, ANCHOR_ON_LAPTOP), null)
  assert.equal(resolvePetOverlayBounds(undefined, LAPTOP, ANCHOR_ON_LAPTOP), null)
  assert.equal(
    resolvePetOverlayBounds({ x: NaN, y: 0, width: 100, height: 100 }, LAPTOP, ANCHOR_ON_LAPTOP),
    null
  )
  assert.equal(
    resolvePetOverlayBounds({ x: 0, y: 0, width: 'wide', height: 100 }, LAPTOP, ANCHOR_ON_LAPTOP),
    null
  )
})

test('resolvePetOverlayBounds returns requested unchanged with no displays to validate against', () => {
  assert.equal(resolvePetOverlayBounds(PET_BOUNDS, null, ANCHOR_ON_LAPTOP), PET_BOUNDS)
  assert.equal(resolvePetOverlayBounds(PET_BOUNDS, [], ANCHOR_ON_LAPTOP), PET_BOUNDS)
})

// ─── on-screen trust ───────────────────────────────────────────────────────

test('an on-screen saved spot is used as-is', () => {
  assert.deepEqual(resolvePetOverlayBounds(PET_BOUNDS, LAPTOP, ANCHOR_ON_LAPTOP), PET_BOUNDS)
})

test('a spot on any connected display is used as-is, even if not the anchor display', () => {
  const onExternal = { x: 1400, y: 200, width: 300, height: 400 }

  assert.deepEqual(resolvePetOverlayBounds(onExternal, LAPTOP_PLUS_EXTERNAL, ANCHOR_ON_LAPTOP), onExternal)
})

test('a spot with only a sliver of overlap (below MIN_VISIBLE) is treated as off-screen', () => {
  // 300px-wide window parked so only 30px of it is inside the laptop panel —
  // less than the 48px MIN_VISIBLE the main window's restore path trusts.
  const sliver = { x: -270, y: 150, width: 300, height: 400 }

  const resolved = resolvePetOverlayBounds(sliver, LAPTOP, ANCHOR_ON_LAPTOP)

  assert.notDeepEqual(resolved, sliver)
  assert.equal(resolved.x, Math.round((1366 - 300) / 2))
  assert.equal(resolved.y, Math.round((728 - 400) / 2))
})

// ─── off-screen re-home ────────────────────────────────────────────────────

test('a spot remembered on a since-unplugged external monitor is re-centered on the anchor display', () => {
  // Saved while the external was connected (x 1366+); only the laptop remains.
  const stale = { x: 1500, y: 400, width: 300, height: 400 }

  const resolved = resolvePetOverlayBounds(stale, LAPTOP, ANCHOR_ON_LAPTOP)

  assert.deepEqual(resolved, {
    x: Math.round((1366 - 300) / 2),
    y: Math.round((728 - 400) / 2),
    width: 300,
    height: 400
  })
})

test('re-home centers on the display holding the anchor (main window), not the primary', () => {
  const stale = { x: -800, y: 400, width: 300, height: 400 }

  const resolved = resolvePetOverlayBounds(stale, LAPTOP_PLUS_EXTERNAL, ANCHOR_ON_EXTERNAL)

  assert.deepEqual(resolved, {
    x: Math.round(1366 + (1920 - 300) / 2),
    y: Math.round((1040 - 400) / 2),
    width: 300,
    height: 400
  })
})

test('re-home falls back to the primary display when the anchor is missing', () => {
  const stale = { x: 1500, y: 400, width: 300, height: 400 }

  const resolved = resolvePetOverlayBounds(stale, LAPTOP, null)

  assert.deepEqual(resolved, {
    x: Math.round((1366 - 300) / 2),
    y: Math.round((728 - 400) / 2),
    width: 300,
    height: 400
  })
})

test('re-homed size is capped to the target work area', () => {
  const huge = { x: 1500, y: 400, width: 2000, height: 1500 }

  const resolved = resolvePetOverlayBounds(huge, LAPTOP, ANCHOR_ON_LAPTOP)

  assert.deepEqual(resolved, {
    x: 0,
    y: 0,
    width: 1366,
    height: 728
  })
})

// ─── live re-home (display unplugged while the pet is popped out) ──────────

test('a still-on-screen pet is left exactly where it is', () => {
  assert.deepEqual(resolvePetOverlayBounds(PET_BOUNDS, LAPTOP, ANCHOR_ON_LAPTOP), PET_BOUNDS)
})

test('a pet stranded by an unplugged display is pulled back onto the anchor display', () => {
  const stranded = { x: 1500, y: 400, width: 300, height: 400 }

  const resolved = resolvePetOverlayBounds(stranded, LAPTOP, ANCHOR_ON_LAPTOP)

  assert.deepEqual(resolved, {
    x: Math.round((1366 - 300) / 2),
    y: Math.round((728 - 400) / 2),
    width: 300,
    height: 400
  })
})
