import { describe, expect, it } from 'vitest'

import { fanOffsets } from './fan-menu'

const STEP = 28

const dist = (a: { x: number; y: number }, b: { x: number; y: number }) => Math.hypot(a.x - b.x, a.y - b.y)

describe('fanOffsets', () => {
  it('stacks a column straight up, one step apart', () => {
    expect(fanOffsets('vertical', 3, STEP)).toEqual([
      { x: 0, y: -STEP },
      { x: 0, y: -2 * STEP },
      { x: 0, y: -3 * STEP }
    ])
  })

  it('splits a row around the hub in reading order, extra one on the right', () => {
    expect(fanOffsets('horizontal', 2, STEP)).toEqual([
      { x: -STEP, y: 0 },
      { x: STEP, y: 0 }
    ])
    expect(fanOffsets('horizontal', 3, STEP)).toEqual([
      { x: -STEP, y: 0 },
      { x: STEP, y: 0 },
      { x: 2 * STEP, y: 0 }
    ])
    expect(fanOffsets('horizontal', 4, STEP).map(p => p.x)).toEqual([-2 * STEP, -STEP, STEP, 2 * STEP])
  })

  // Neighbours never touch in any direction: the whole point of the step.
  it('keeps arc neighbours one step apart and clear of the hub', () => {
    for (const count of [1, 2, 3, 5, 8]) {
      const points = fanOffsets('arc', count, STEP)
      const hub = { x: 0, y: 0 }

      expect(points).toHaveLength(count)

      for (const p of points) {
        expect(p.y).toBeLessThan(0)
        expect(dist(p, hub)).toBeGreaterThanOrEqual(STEP - 1e-6)
      }

      for (let i = 1; i < count; i++) {
        expect(dist(points[i - 1], points[i])).toBeCloseTo(STEP, 6)
      }
    }
  })

  it('centres the arc over the hub', () => {
    const points = fanOffsets('arc', 4, STEP)

    expect(points[0].x).toBeCloseTo(-points[3].x, 6)
    expect(points[1].x).toBeCloseTo(-points[2].x, 6)
  })
})
