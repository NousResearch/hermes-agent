import { describe, expect, it } from 'vitest'

import { mapPool } from './pool'

describe('mapPool', () => {
  it('preserves input order regardless of completion order', async () => {
    const out = await mapPool([30, 10, 20], 3, async ms => {
      await new Promise(r => setTimeout(r, ms))

      return ms
    })

    expect(out).toEqual([30, 10, 20])
  })

  it('never exceeds the concurrency limit', async () => {
    let active = 0
    let peak = 0

    await mapPool([...Array(10).keys()], 3, async () => {
      active += 1
      peak = Math.max(peak, active)
      await new Promise(r => setTimeout(r, 5))
      active -= 1
    })

    expect(peak).toBeLessThanOrEqual(3)
  })
})
