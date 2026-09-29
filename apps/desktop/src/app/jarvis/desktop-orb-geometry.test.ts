import { describe, expect, it } from 'vitest'

import {
  clampOrbScale,
  clampToArea,
  ORB_SCALE_MAX,
  ORB_SCALE_MIN,
  orbWindowSize,
  resizeAroundCenter,
  scaleFromDrag,
  snapToEdges
} from './desktop-orb-geometry'

const area = { height: 1000, width: 1600, x: 0, y: 0 }

describe('orb geometry', () => {
  it('keeps the scale in range and falls back to normal size for nonsense', () => {
    expect(clampOrbScale(9)).toBe(ORB_SCALE_MAX)
    expect(clampOrbScale(0.01)).toBe(ORB_SCALE_MIN)
    expect(clampOrbScale(Number.NaN)).toBe(1)
    expect(orbWindowSize(2).width).toBe(orbWindowSize(ORB_SCALE_MAX).width)
  })

  it('grows and shrinks around the same centre', () => {
    const before = { height: 360, width: 320, x: 500, y: 200 }
    const after = resizeAroundCenter(before, orbWindowSize(1.5))

    expect(after.x + after.width / 2).toBeCloseTo(before.x + before.width / 2, 0)
    expect(after.y + after.height / 2).toBeCloseTo(before.y + before.height / 2, 0)
    expect(after.width).toBeGreaterThan(before.width)
  })

  it('never lets the orb be dragged or resized out of reach', () => {
    const lost = clampToArea({ height: 360, width: 320, x: 5000, y: -900 }, area)

    expect(lost.x).toBeLessThan(area.width)
    expect(lost.y + lost.height).toBeGreaterThan(0)
    // Inside the area nothing moves.
    expect(clampToArea({ height: 360, width: 320, x: 100, y: 100 }, area)).toMatchObject({ x: 100, y: 100 })
  })

  it('snaps to a nearby edge with a margin, and only then reports it', () => {
    const near = snapToEdges({ height: 360, width: 320, x: 14, y: 400 }, area)

    expect(near.snapped).toBe(true)
    expect(near.bounds.x).toBe(12)
    expect(near.bounds.y).toBe(400)

    const nearRight = snapToEdges({ height: 360, width: 320, x: 1600 - 320 - 10, y: 400 }, area)
    expect(nearRight.bounds.x).toBe(1600 - 320 - 12)

    expect(snapToEdges({ height: 360, width: 320, x: 600, y: 300 }, area).snapped).toBe(false)
  })

  it('scales with how far the grip is dragged from the centre', () => {
    expect(scaleFromDrag(1, 100, 150)).toBe(1.5)
    expect(scaleFromDrag(1, 100, 10)).toBe(ORB_SCALE_MIN)
    expect(scaleFromDrag(1.2, 0, 50)).toBe(1.2)
  })
})
