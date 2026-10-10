import assert from 'node:assert/strict'

import { test } from 'vitest'

import { sanitizeHudState } from './hud-state'

test('a remembered snapshot keeps its geometry and reports the shipped pin state', () => {
  assert.deepEqual(sanitizeHudState({ x: 100, y: 500, width: 620, height: 320 }), {
    alwaysOnTop: true,
    height: 320,
    width: 620,
    x: 100,
    y: 500
  })
})

test('the pin preference is strict, and survives a round trip', () => {
  assert.equal(sanitizeHudState({ x: 0, y: 0, width: 620, height: 320, alwaysOnTop: false })!.alwaysOnTop, false)
  assert.equal(sanitizeHudState({ x: 0, y: 0, width: 620, height: 320, alwaysOnTop: true })!.alwaysOnTop, true)
  // Anything but a real `true` reads as unpinned — a string from a hand-edited
  // file must not silently re-pin a window the user released.
  assert.equal(sanitizeHudState({ x: 0, y: 0, width: 620, height: 320, alwaysOnTop: 'yes' })!.alwaysOnTop, false)
})

test('structurally broken snapshots are rejected outright', () => {
  for (const raw of [
    null,
    'hud',
    7,
    {},
    { x: 0, y: 0, width: 620 },
    { x: 0, y: 0, width: Number.NaN, height: 320 },
    { x: Number.POSITIVE_INFINITY, y: 0, width: 620, height: 320 },
    // Below the HUD's minimums: not a bar the user could work in.
    { x: 0, y: 0, width: 379, height: 320 },
    { x: 0, y: 0, width: 620, height: 159 }
  ]) {
    assert.equal(sanitizeHudState(raw), null)
  }
})
