import { describe, expect, it } from 'vitest'

import { createStreamTpsCounter } from './stream-tps'

describe('createStreamTpsCounter', () => {
  it('returns undefined while fewer than 3 samples are accumulated', () => {
    const counter = createStreamTpsCounter()

    // First call always emits (lastEmit starts at 0): 1 sample => undefined.
    expect(counter.accumulate(1_000)).toBeUndefined()
    // 1 s later the throttle re-emits with 2 samples => still undefined.
    expect(counter.accumulate(2_000)).toBeUndefined()
  })

  it('returns Math.round((count/2)*10)/10 once >= 3 samples are inside the 2s window', () => {
    // count = 3 => Math.round((3/2)*10)/10 = Math.round(15)/10 = 1.5
    const three = createStreamTpsCounter()
    expect(three.accumulate(1_000)).toBeUndefined() // <3
    expect(three.accumulate(1_100)).toBeUndefined() // throttle holds (2 samples)
    expect(three.accumulate(2_110)).toBe(1.5) // 3 samples in window

    // count = 4 => Math.round((4/2)*10)/10 = Math.round(20)/10 = 2
    const four = createStreamTpsCounter()
    expect(four.accumulate(1_000)).toBeUndefined()
    expect(four.accumulate(1_100)).toBeUndefined() // hold
    expect(four.accumulate(1_200)).toBeUndefined() // hold
    expect(four.accumulate(2_160)).toBe(2) // 4 samples in window
  })

  it('ignores timestamps older than 2s (rolling window)', () => {
    const counter = createStreamTpsCounter()

    expect(counter.accumulate(1_000)).toBeUndefined() // 1 sample
    expect(counter.accumulate(2_000)).toBeUndefined() // 2 samples
    expect(counter.accumulate(3_000)).toBe(1.5) // 3 samples

    // Jump > 2 s ahead: every previous timestamp is now outside the window, so
    // the count falls back below the minimum and the rate reverts to undefined.
    expect(counter.accumulate(5_500)).toBeUndefined()
  })

  it('throttles: does not re-emit before ~1s has elapsed', () => {
    const counter = createStreamTpsCounter()

    expect(counter.accumulate(1_000)).toBeUndefined() // emits, 1 sample

    // 5 rapid deltas within the same second: never re-emit, even though the
    // window already holds > 3 samples (which WOULD emit if not throttled).
    expect(counter.accumulate(1_100)).toBeUndefined()
    expect(counter.accumulate(1_200)).toBeUndefined()
    expect(counter.accumulate(1_300)).toBeUndefined()
    expect(counter.accumulate(1_400)).toBeUndefined()
    expect(counter.accumulate(1_500)).toBeUndefined() // throttle still active

    // Once ~1 s has elapsed since the last emit, it emits again with the
    // full 7-sample window: Math.round((7/2)*10)/10 = Math.round(35)/10 = 3.5.
    expect(counter.accumulate(2_160)).toBe(3.5)
  })

  it('reset() clears the state so the next accumulate starts from zero', () => {
    const counter = createStreamTpsCounter()

    expect(counter.accumulate(1_000)).toBeUndefined()
    expect(counter.accumulate(1_100)).toBeUndefined()
    expect(counter.accumulate(2_110)).toBe(1.5) // builds up a 3-sample rate

    counter.reset()

    // After reset we are back below the 3-sample minimum...
    expect(counter.accumulate(5_000)).toBeUndefined()
    // ...and can build the rate up again from scratch.
    expect(counter.accumulate(5_100)).toBeUndefined()
    expect(counter.accumulate(6_110)).toBe(1.5)
  })
})
