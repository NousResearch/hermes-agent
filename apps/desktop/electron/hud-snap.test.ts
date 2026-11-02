import assert from 'node:assert/strict'

import { test } from 'vitest'

import { clampHudOrigin, snapHudBounds, windowOriginForCursorAnchor } from './hud-snap'

const WORK = { x: 0, y: 25, width: 1440, height: 875 }
const SIZE = { width: 620, height: 320 }

test('anchor under cursor at zoom 1', () => {
  assert.deepEqual(windowOriginForCursorAnchor({ x: 500, y: 400 }, { x: 310, y: 48 }, 1), { x: 190, y: 352 })
})

test('anchor scales with page zoom', () => {
  assert.deepEqual(windowOriginForCursorAnchor({ x: 500, y: 400 }, { x: 100, y: 50 }, 0.9), {
    x: Math.round(500 - 100 * 0.9),
    y: Math.round(400 - 50 * 0.9)
  })
})

test('clamp keeps a sliver visible when the anchor would park off-screen', () => {
  const origin = windowOriginForCursorAnchor({ x: 10, y: 30 }, { x: 310, y: 48 }, 1)
  const clamped = clampHudOrigin(origin, SIZE, WORK)

  assert.ok(clamped.x >= WORK.x + 40 - SIZE.width)
  assert.ok(clamped.y >= WORK.y + 40 - SIZE.height)
})

test('snapHudBounds composes origin + clamp', () => {
  const point = snapHudBounds({ x: 720, y: 450 }, { x: 310, y: 48 }, SIZE, 1, WORK)

  assert.equal(point.x, 410)
  assert.equal(point.y, 402)
})
