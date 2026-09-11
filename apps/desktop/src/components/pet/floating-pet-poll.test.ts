import { describe, expect, it } from 'vitest'

import { petInfoPollIntervalMs } from './pet-info-poll'

describe('petInfoPollIntervalMs', () => {
  it('uses the slow backstop on event-capable backends (active or not)', () => {
    expect(petInfoPollIntervalMs(true, false)).toBe(15_000)
    expect(petInfoPollIntervalMs(true, true)).toBe(15_000)
  })

  it('keeps the legacy fast-while-inactive cadence without change events', () => {
    expect(petInfoPollIntervalMs(false, false)).toBe(3_000)
    expect(petInfoPollIntervalMs(false, true)).toBe(15_000)
  })
})
