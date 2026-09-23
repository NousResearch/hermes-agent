import assert from 'node:assert/strict'

import { test } from 'vitest'

import { clampHudOrigin, dockHudToNearestEdge, snapHudBounds, windowOriginForCursorAnchor } from './hud-snap'

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

test('dock flushes to the work-area bottom when within threshold', () => {
  const origin = { x: 400, y: WORK.y + WORK.height - SIZE.height - 12 }
  const docked = dockHudToNearestEdge(origin, SIZE, WORK, 24)

  assert.equal(docked.y, WORK.y + WORK.height - SIZE.height)
  assert.equal(docked.x, 400)
})

test('dock fail-open keeps a free placement 40px from the bottom', () => {
  const origin2 = { x: 400, y: WORK.y + WORK.height - SIZE.height - 40 }

  assert.deepEqual(dockHudToNearestEdge(origin2, SIZE, WORK, 24), origin2)
})

test('dock flushes to the work-area top when within threshold', () => {
  const origin = { x: 400, y: WORK.y + 12 }
  const docked = dockHudToNearestEdge(origin, SIZE, WORK, 24)

  assert.equal(docked.y, WORK.y)
  assert.equal(docked.x, 400)
})

test('dock flushes to the work-area left when within threshold', () => {
  const origin = { x: WORK.x + 12, y: 200 }
  const docked = dockHudToNearestEdge(origin, SIZE, WORK, 24)

  assert.equal(docked.x, WORK.x)
  assert.equal(docked.y, 200)
})

test('dock flushes to the work-area right when within threshold', () => {
  const origin = { x: WORK.x + WORK.width - SIZE.width - 12, y: 200 }
  const docked = dockHudToNearestEdge(origin, SIZE, WORK, 24)

  assert.equal(docked.x, WORK.x + WORK.width - SIZE.width)
  assert.equal(docked.y, 200)
})

test('dock fail-open returns the original origin when workArea is not finite', () => {
  const origin = { x: 400, y: WORK.y + WORK.height - SIZE.height - 12 }

  assert.deepEqual(
    dockHudToNearestEdge(origin, SIZE, { x: 0, y: 25, width: Number.NaN, height: 875 }, 24),
    origin
  )
  assert.deepEqual(
    dockHudToNearestEdge(origin, SIZE, { x: 0, y: Number.POSITIVE_INFINITY, width: 1440, height: 875 }, 24),
    origin
  )
})

test('dock prefers the vertical edge when a corner is equally close', () => {
  const origin = {
    x: WORK.x + WORK.width - SIZE.width - 12,
    y: WORK.y + WORK.height - SIZE.height - 12
  }
  const docked = dockHudToNearestEdge(origin, SIZE, WORK, 24)

  assert.equal(docked.y, WORK.y + WORK.height - SIZE.height)
  assert.equal(docked.x, origin.x)
})

test('snapHudBounds then dock flushes when the cursor parks within 24px of the bottom', () => {
  const anchor = { x: 310, y: 48 }
  const cursor = { x: 710, y: WORK.y + WORK.height - SIZE.height - 12 + anchor.y }
  const snapped = snapHudBounds(cursor, anchor, SIZE, 1, WORK)
  const docked = dockHudToNearestEdge(snapped, SIZE, WORK, 24)

  assert.equal(snapped.x, 400)
  assert.equal(snapped.y, WORK.y + WORK.height - SIZE.height - 12)
  assert.equal(docked.y, WORK.y + WORK.height - SIZE.height)
  assert.equal(docked.x, 400)
})
