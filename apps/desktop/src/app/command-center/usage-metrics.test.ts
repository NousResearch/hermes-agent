import { describe, expect, it } from 'vitest'

import { cacheHitSummary } from './usage-metrics'

describe('cacheHitSummary', () => {
  it('computes the rate from tokens rather than averaging daily percentages', () => {
    const summary = cacheHitSummary({ cache_read_tokens: 1_000_000, cache_write_tokens: 0, input_tokens: 1_000 })

    expect(summary).toMatchObject({ cachedTokens: 1_000_000, promptTokens: 1_001_000 })
    expect(summary?.hitRate).toBeCloseTo(99.9, 1)
  })

  it('keeps cache rate absent when the bucket has no prompt tokens', () => {
    expect(cacheHitSummary({ cache_read_tokens: 0, cache_write_tokens: 0, input_tokens: 0 })).toBeNull()
  })

  it('does not turn missing cache accounting into a false zero-percent point', () => {
    expect(cacheHitSummary({ cache_read_tokens: 0, cache_write_tokens: 0, input_tokens: 500 })).toBeNull()
  })
})
