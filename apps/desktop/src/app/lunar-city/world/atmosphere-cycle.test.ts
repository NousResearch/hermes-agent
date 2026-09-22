import { describe, expect, it } from 'vitest'

import { createAtmosphereClock } from './atmosphere-cycle'

describe('local atmosphere presentation clock', () => {
  it('holds the sky exactly under reduced motion and does not catch up after a suspended tab', () => {
    const clock = createAtmosphereClock()
    const start = clock.advance(0, false)
    expect(clock.advance(60_000, true)).toEqual(start)
    const resumed = clock.advance(60_000, false)
    expect(resumed.phase - start.phase).toBeCloseTo(1000 / (20 * 60 * 1000))
  })
  it('moves through daylight and night while keeping working surfaces illuminated', () => {
    const clock = createAtmosphereClock(4000)
    const day = clock.advance(0, false)
    clock.advance(1000, false)
    const night = clock.advance(1000, false)
    expect(day.daylight).toBeGreaterThan(.9)
    expect(night.daylight).toBe(0)
    expect(night.keyIntensity).toBeGreaterThan(0)
    expect(night.fillIntensity).toBeGreaterThan(.4)
    expect(clock.advance(Number.NaN, false)).toEqual(night)
  })
})
