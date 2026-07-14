import assert from 'node:assert/strict'

import { test } from 'vitest'

import { clampPetOverlayBounds, type PetOverlayDisplay } from './pet-overlay-geometry'

const PRIMARY: PetOverlayDisplay = { workArea: { height: 1040, width: 1920, x: 0, y: 0 } }
const LEFT: PetOverlayDisplay = { workArea: { height: 900, width: 1440, x: -1440, y: 0 } }

test('keeps valid in-work-area bounds unchanged', () => {
  assert.deepEqual(clampPetOverlayBounds({ height: 300, width: 360, x: 100, y: 200 }, [PRIMARY]), {
    height: 300,
    width: 360,
    x: 100,
    y: 200
  })
})

test('clamps a partially offscreen panel fully inside its intersecting display', () => {
  assert.deepEqual(clampPetOverlayBounds({ height: 300, width: 360, x: 1800, y: 900 }, [PRIMARY]), {
    height: 300,
    width: 360,
    x: 1560,
    y: 740
  })
})

test('moves a completely orphaned monitor position to the nearest display', () => {
  assert.deepEqual(clampPetOverlayBounds({ height: 300, width: 360, x: 2500, y: 200 }, [LEFT, PRIMARY]), {
    height: 300,
    width: 360,
    x: 1560,
    y: 200
  })
})

test('supports negative-coordinate displays', () => {
  assert.deepEqual(clampPetOverlayBounds({ height: 300, width: 360, x: -1500, y: 100 }, [LEFT, PRIMARY]), {
    height: 300,
    width: 360,
    x: -1440,
    y: 100
  })
})

test('shrinks oversized panels to the selected work area', () => {
  assert.deepEqual(clampPetOverlayBounds({ height: 1500, width: 2500, x: 100, y: 100 }, [PRIMARY]), {
    height: 1040,
    width: 1920,
    x: 0,
    y: 0
  })
})

test('sanitizes invalid numeric input and preserves a safe fallback without displays', () => {
  assert.deepEqual(
    clampPetOverlayBounds(
      { height: Number.NaN, width: Number.NEGATIVE_INFINITY, x: Number.POSITIVE_INFINITY, y: -12.6 },
      []
    ),
    { height: 80, width: 80, x: 0, y: -13 }
  )
})

test('breaks equal-distance display ties by input order', () => {
  const right: PetOverlayDisplay = { workArea: { height: 800, width: 1000, x: 0, y: 0 } }
  const left: PetOverlayDisplay = { workArea: { height: 800, width: 1000, x: -1000, y: 0 } }

  assert.equal(clampPetOverlayBounds({ height: 100, width: 100, x: -50, y: 900 }, [right, left]).x, 0)
  assert.equal(clampPetOverlayBounds({ height: 100, width: 100, x: -50, y: 900 }, [left, right]).x, -100)
})

test('reclamps stale bounds after a display is removed or its work area shrinks', () => {
  const stale = { height: 700, width: 1200, x: 2000, y: 300 }
  const remaining = [{ workArea: { height: 728, width: 1366, x: 0, y: 0 } }]

  assert.deepEqual(clampPetOverlayBounds(stale, remaining), {
    height: 700,
    width: 1200,
    x: 166,
    y: 28
  })
})
