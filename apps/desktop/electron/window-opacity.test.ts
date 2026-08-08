/**
 * Unit tests for the pure translucency helpers: the intensity -> opacity ramp
 * the window options and the runtime IPC toggle share, its endpoints, and the
 * clamping that keeps a corrupt translucency.json from reaching setOpacity.
 */

import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  clampIntensity,
  TRANSLUCENCY_CURVE,
  TRANSLUCENCY_MAX,
  TRANSLUCENCY_MIN,
  TRANSLUCENCY_OPACITY_FLOOR,
  windowOpacity
} from './window-opacity'

/** The linear ramp this curve replaced. Endpoints must still agree with it. */
const legacyOpacity = (intensity: number) => 1 - (intensity / 100) * 0.7

test('lever bounds and floor stay stable so persisted settings survive upgrades', () => {
  assert.equal(TRANSLUCENCY_MIN, 0)
  assert.equal(TRANSLUCENCY_MAX, 100)
  assert.equal(TRANSLUCENCY_OPACITY_FLOOR, 0.3)
})

test('endpoints are unchanged from the linear ramp', () => {
  assert.equal(windowOpacity(TRANSLUCENCY_MIN), legacyOpacity(0))
  assert.equal(windowOpacity(TRANSLUCENCY_MAX), legacyOpacity(100))
})

test('off is fully opaque and max is exactly the floor', () => {
  assert.equal(windowOpacity(0), 1)
  assert.equal(windowOpacity(100), 1 - (1 - TRANSLUCENCY_OPACITY_FLOOR))
})

test('opacity decreases monotonically and never sinks below the floor', () => {
  let previous = windowOpacity(TRANSLUCENCY_MIN)

  for (let intensity = TRANSLUCENCY_MIN + 1; intensity <= TRANSLUCENCY_MAX; intensity += 1) {
    const opacity = windowOpacity(intensity)

    assert.ok(opacity < previous, `expected ${intensity} to be more see-through than ${intensity - 1}`)
    assert.ok(opacity >= TRANSLUCENCY_OPACITY_FLOOR, `expected ${intensity} to stay at or above the floor`)

    previous = opacity
  }
})

// The bug this module fixes: setOpacity fades text too, so anything under ~0.95
// is where reading gets hard. A linear ramp put that boundary at intensity 7,
// leaving almost the whole lever unusable.
test('the readable band covers a useful stretch of the lever, not just its first few percent', () => {
  const readable = []

  for (let intensity = TRANSLUCENCY_MIN; intensity <= TRANSLUCENCY_MAX; intensity += 1) {
    if (windowOpacity(intensity) >= 0.95) {
      readable.push(intensity)
    }
  }

  assert.ok(
    readable.length > 20,
    `expected more than 20 readable settings, got ${readable.length} (max ${readable[readable.length - 1]})`
  )
})

test('a curved ramp keeps fine control near the opaque end', () => {
  // Linear gave 0.965 here — a visible jump off "off" on the very first step.
  assert.ok(windowOpacity(5) > 0.99, `expected 5% to stay near-opaque, got ${windowOpacity(5)}`)
  assert.ok(windowOpacity(1) > legacyOpacity(1))
  assert.ok(TRANSLUCENCY_CURVE > 1)
})

test('clampIntensity rejects garbage and enforces bounds', () => {
  assert.equal(clampIntensity(NaN), TRANSLUCENCY_MIN)
  assert.equal(clampIntensity(Infinity), TRANSLUCENCY_MIN)
  assert.equal(clampIntensity(undefined), TRANSLUCENCY_MIN)
  assert.equal(clampIntensity(null), TRANSLUCENCY_MIN)
  assert.equal(clampIntensity('nope'), TRANSLUCENCY_MIN)
  assert.equal(clampIntensity(-40), TRANSLUCENCY_MIN)
  assert.equal(clampIntensity(240), TRANSLUCENCY_MAX)
  assert.equal(clampIntensity('35'), 35)
  assert.equal(clampIntensity(35.6), 36)
})

test('windowOpacity clamps before mapping so corrupt state cannot escape the range', () => {
  assert.equal(windowOpacity(-40), 1)
  assert.equal(windowOpacity(240), windowOpacity(TRANSLUCENCY_MAX))
  assert.equal(windowOpacity('nope'), 1)
})
