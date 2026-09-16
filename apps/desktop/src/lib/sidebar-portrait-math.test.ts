import { describe, expect, it } from 'vitest'

import { PORTRAIT_REVEAL_MS, shapeLuminance, warmColor } from './sidebar-portrait-math'

describe('warmColor', () => {
  it('returns the darkest gradient stop for zero intensity', () => {
    expect(warmColor(0)).toEqual([23, 8, 6])
  })

  it('returns the brightest gradient stop for full intensity', () => {
    expect(warmColor(255)).toEqual([255, 215, 163])
  })

  it('interpolates between adjacent stops', () => {
    const [r, g, b] = warmColor(128)

    expect(r).toBeGreaterThan(23)
    expect(r).toBeLessThanOrEqual(255)
    expect(g).toBeGreaterThan(8)
    expect(b).toBeGreaterThanOrEqual(6)
    // Warm palette: red always dominates blue.
    expect(r).toBeGreaterThan(b)
  })

  it('clamps out-of-range input instead of producing invalid colors', () => {
    expect(warmColor(-10)).toEqual(warmColor(0))
    expect(warmColor(500)).toEqual(warmColor(255))
  })
})

describe('shapeLuminance', () => {
  it('reveals nothing at elapsed 0', () => {
    expect(shapeLuminance(255, 0)).toBe(0)
  })

  it('reaches full gamma-shaped intensity once the reveal completes', () => {
    expect(shapeLuminance(255, PORTRAIT_REVEAL_MS)).toBe(255)
    expect(shapeLuminance(255, PORTRAIT_REVEAL_MS * 2)).toBe(255)
  })

  it('stays proportional to the reveal progress mid-way', () => {
    const mid = shapeLuminance(255, PORTRAIT_REVEAL_MS / 2)

    expect(mid).toBeGreaterThan(100)
    expect(mid).toBeLessThan(200)
  })
})
