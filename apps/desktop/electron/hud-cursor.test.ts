/**
 * Unit tests for the HUD's Linux cursor conversion. These cover the two things
 * that decide whether the bar is clickable: landing on the right point, and
 * admitting when the cursor has left.
 */

import assert from 'node:assert/strict'

import { test } from 'vitest'

import { cursorPointInWindow } from './hud-cursor'

// A HUD parked 300px right and 200px down, 620×320 as it is created.
const BOUNDS = { x: 300, y: 200, width: 620, height: 320 }

test('screen cursor becomes a point relative to the window', () => {
  assert.deepEqual(cursorPointInWindow({ x: 300, y: 200 }, BOUNDS, 1), { x: 0, y: 0 })
  assert.deepEqual(cursorPointInWindow({ x: 460, y: 240 }, BOUNDS, 1), { x: 160, y: 40 })
})

test('a zoomed page reports CSS pixels, not device-independent ones', () => {
  // At 0.9 the page is wider than the window in CSS pixels, so a point half way
  // across the window is further than half way across the document.
  assert.deepEqual(cursorPointInWindow({ x: 610, y: 380 }, BOUNDS, 0.9), {
    x: 310 / 0.9,
    y: 180 / 0.9
  })
})

test('a cursor outside the window is reported as lost', () => {
  for (const outside of [
    { x: 299, y: 300 },
    { x: 500, y: 199 },
    { x: 920, y: 300 },
    { x: 500, y: 520 }
  ]) {
    assert.equal(cursorPointInWindow(outside, BOUNDS, 1), null)
  }
})

test('the far edges belong to the window on entry, not on exit', () => {
  assert.deepEqual(cursorPointInWindow({ x: 919, y: 519 }, BOUNDS, 1), { x: 619, y: 319 })
  assert.equal(cursorPointInWindow({ x: 920, y: 520 }, BOUNDS, 1), null)
})

test('a nonsense zoom factor is treated as unzoomed rather than poisoning the point', () => {
  for (const bad of [0, -1, Number.NaN, Number.POSITIVE_INFINITY]) {
    assert.deepEqual(cursorPointInWindow({ x: 460, y: 240 }, BOUNDS, bad), { x: 160, y: 40 })
  }
})
