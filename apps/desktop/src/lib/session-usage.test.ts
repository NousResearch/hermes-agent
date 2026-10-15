import { describe, expect, it } from 'vitest'

import { usageFromTick } from '@/lib/session-usage'
import { cacheHitLabel, tokensPerSecondLabel } from '@/lib/statusbar'

const base = { calls: 0, input: 0, output: 0, total: 0 }

describe('canonical session usage projection', () => {
  it('turns owner counters into visible cache and throughput labels', () => {
    const usage = {
      ...base,
      ...usageFromTick({ api_calls: 2, cache_read_tokens: 870, prompt_tokens: 1000, tokens_per_s: 41.6 })
    }

    expect(cacheHitLabel(usage)).toBe('87%')
    expect(tokensPerSecondLabel(usage)).toBe('42 t/s')
  })

  it('preserves a legacy partial usage payload without inventing metrics', () => {
    const usage = { input: 500, cache_hit_pct: 80, avg_tps: 30 }

    expect(usageFromTick({ usage })).toBe(usage)
    expect(usageFromTick({ api_calls: 0 })).toEqual(base)
    expect(usageFromTick(undefined)).toBeUndefined()
    expect(usageFromTick({})).toBeUndefined()
  })

  it('does not paint non-finite throughput or divide by an absent prompt count', () => {
    const usage = usageFromTick({ api_calls: 1, cache_read_tokens: 100, tokens_per_s: Number.NaN })

    expect(usage?.cache_hit_pct).toBeUndefined()
    expect(usage?.avg_tps).toBeUndefined()
  })
})
