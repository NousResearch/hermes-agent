import { describe, expect, it } from 'vitest'

import { cacheHitLabel, sessionCostLabel, sessionTokensLabel, tokensPerSecondLabel } from '@/lib/statusbar'

const base = { calls: 0, input: 0, output: 0, total: 0 }

describe('statusbar usage readouts', () => {
  it('paints the backend cache-hit and throughput fields, and stays blank when they are absent', () => {
    // The backend omits both fields (rather than sending 0) when it has no data
    // — a provider with no cache reads, or a session before its first call.
    expect(cacheHitLabel(base)).toBe('')
    expect(tokensPerSecondLabel(base)).toBe('')

    expect(cacheHitLabel({ ...base, cache_hit_pct: 87 })).toBe('87%')
    expect(tokensPerSecondLabel({ ...base, avg_tps: 41.6 })).toBe('42 t/s')
  })
})

describe('session token and spend readouts', () => {
  it('reads cumulative prompt/completion tokens and stays blank before the first call', () => {
    expect(sessionTokensLabel(base)).toBe('')
    expect(sessionTokensLabel({ ...base, input: 12_345, output: 4_200 })).toBe('↓12.3k ↑4.2k')
  })

  it('keeps three or four decimals for sub-yuan spend and marks an estimate with ~', () => {
    // A route with no pricing entry sends no cost at all — blank, never ¥0.0000.
    expect(sessionCostLabel(base)).toBe('')
    expect(sessionCostLabel({ ...base, cost_status: 'estimated', cost_usd: 0.0027 })).toBe('~¥0.019')
    expect(sessionCostLabel({ ...base, cost_status: 'actual', cost_usd: 0.0312 })).toBe('¥0.225')
    expect(sessionCostLabel({ ...base, cost_status: 'estimated', cost_usd: 1.5 })).toBe('~¥10.80')
  })

  it('labels a subscription session instead of pricing it', () => {
    expect(sessionCostLabel({ ...base, cost_status: 'included' }, 'included')).toBe('included')
  })
})
