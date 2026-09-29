import { describe, expect, it } from 'vitest'

import { AURA_BARS, auraTargets } from './voice-aura'

describe('auraTargets', () => {
  it('draws nothing for silence, whatever the phase', () => {
    for (const phase of [0, 1.7, 40]) {
      expect(auraTargets(0, phase).every(value => value === 0)).toBe(true)
    }
  })

  it('mirrors left and right, and never exceeds the measured level', () => {
    const level = 0.8
    const bars = auraTargets(level, 2.3)

    expect(bars).toHaveLength(AURA_BARS)

    for (let i = 1; i < AURA_BARS / 2; i += 1) {
      expect(bars[i]).toBeCloseTo(bars[AURA_BARS - i]!, 10)
    }

    expect(Math.max(...bars)).toBeLessThanOrEqual(level)
    expect(Math.min(...bars)).toBeGreaterThanOrEqual(0)
  })
})
