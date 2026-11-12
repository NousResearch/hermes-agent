import { describe, expect, it } from 'vitest'

import { anchoredRect, clampFloatingRect, floatingPx, reflowRect } from './floating-rect'

const viewport = { width: 1440, height: 900, top: 34 }
const size = { width: 240, height: 180 }

describe('anchoredRect', () => {
  it('spawns in the named corner, inset by the margin', () => {
    expect(anchoredRect('top-right', size, viewport)).toMatchObject({ x: 1188, y: 46 })
    expect(anchoredRect('top-left', size, viewport)).toMatchObject({ x: 12, y: 46 })
    expect(anchoredRect('bottom-right', size, viewport)).toMatchObject({ x: 1188, y: 708 })
    expect(anchoredRect('bottom-left', size, viewport)).toMatchObject({ x: 12, y: 708 })
  })

  it('never spawns over the reserved top chrome', () => {
    for (const anchor of ['top-left', 'top-right', 'bottom-left', 'bottom-right'] as const) {
      expect(anchoredRect(anchor, size, { width: 320, height: 120, top: 34 }).y).toBeGreaterThanOrEqual(34)
    }
  })
})

describe('clampFloatingRect', () => {
  it('keeps a grabbable sliver on screen at either horizontal edge', () => {
    expect(clampFloatingRect({ ...size, x: 5000, y: 100 }, viewport).x).toBe(1392)
    expect(clampFloatingRect({ ...size, x: -5000, y: 100 }, viewport).x).toBe(-192)
  })

  it('hard-bounds the top so the drag handle is always reachable', () => {
    expect(clampFloatingRect({ ...size, x: 100, y: -500 }, viewport).y).toBe(34)
    expect(clampFloatingRect({ ...size, x: 100, y: 5000 }, viewport).y).toBe(852)
  })

  it('pins to the top-left rather than inverting when larger than the viewport', () => {
    const tiny = { width: 200, height: 100, top: 34 }
    const rect = clampFloatingRect({ x: 0, y: 0, width: 400, height: 400 }, tiny)

    expect(rect.x).toBeLessThanOrEqual(tiny.width)
    expect(rect.y).toBe(34)
  })

  it('leaves an in-bounds rect untouched', () => {
    const rect = { ...size, x: 400, y: 200 }

    expect(clampFloatingRect(rect, viewport)).toEqual(rect)
  })
})

describe('reflowRect', () => {
  const shrunk = { width: 1000, height: 700, top: 34 }

  it('rides the right edge when right-anchored', () => {
    const start = anchoredRect('top-right', size, viewport)

    expect(reflowRect(start, 'top-right', viewport, shrunk).x).toBe(748)
  })

  it('rides the bottom edge when bottom-anchored', () => {
    const start = anchoredRect('bottom-left', size, viewport)

    expect(reflowRect(start, 'bottom-left', viewport, shrunk).y).toBe(508)
  })

  it('holds position when left/top-anchored, but still clamps into view', () => {
    const start = { ...size, x: 40, y: 60 }

    expect(reflowRect(start, 'top-left', viewport, shrunk)).toMatchObject({ x: 40, y: 60 })
    expect(reflowRect({ ...size, x: 1300, y: 60 }, 'top-left', viewport, shrunk).x).toBe(952)
  })

  it('is identity for an in-bounds rect when the viewport did not change', () => {
    const rect = { ...size, x: 300, y: 300 }

    expect(reflowRect(rect, 'top-right', viewport, viewport)).toEqual(rect)
  })
})

describe('floatingPx', () => {
  it('accepts authored px strings and raw numbers, falling back otherwise', () => {
    expect(floatingPx('216px', 240)).toBe(216)
    expect(floatingPx(300, 240)).toBe(300)
    expect(floatingPx(undefined, 240)).toBe(240)
    expect(floatingPx('auto', 240)).toBe(240)
    expect(floatingPx(Number.NaN, 240)).toBe(240)
  })
})
