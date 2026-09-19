import { describe, expect, it } from 'vitest'

import { AdaptiveAcousticThreshold } from './voice-acoustics'

describe('AdaptiveAcousticThreshold', () => {
  it('raises both capture and barge-in thresholds with quiet ambient drift', () => {
    const threshold = new AdaptiveAcousticThreshold()
    const initial = threshold.startThreshold

    for (let index = 0; index < 40; index += 1) {
      threshold.observeQuiet(0.04)
    }

    expect(threshold.noiseFloor).toBeGreaterThan(0.01)
    expect(threshold.startThreshold).toBeGreaterThan(initial)
    expect(threshold.endThreshold).toBeLessThan(threshold.startThreshold)
  })

  it('does not learn from speech spikes or exceed the bounded floor', () => {
    const threshold = new AdaptiveAcousticThreshold(0.08)
    threshold.observeQuiet(0.8)
    expect(threshold.noiseFloor).toBe(0.01)

    for (let index = 0; index < 200; index += 1) {
      threshold.observeQuiet(0.07)
    }

    expect(threshold.noiseFloor).toBeLessThanOrEqual(0.08)
  })
})
