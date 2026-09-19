import { describe, expect, it } from 'vitest'

import { cacheHitLabel, timeToFirstTokenLabel, tokensPerSecondLabel } from '@/lib/statusbar'

const base = { calls: 0, input: 0, output: 0, total: 0 }

describe('statusbar usage readouts', () => {
  it('paints the backend cache-hit, throughput, and TTFT fields, and stays blank when they are absent', () => {
    // The backend omits fields (rather than sending 0) when it has no data
    // — a provider with no cache reads, or a session before its first call.
    expect(cacheHitLabel(base)).toBe('')
    expect(tokensPerSecondLabel(base)).toBe('')
    expect(timeToFirstTokenLabel(base)).toBe('')

    expect(cacheHitLabel({ ...base, cache_hit_pct: 87 })).toBe('87%')
    expect(tokensPerSecondLabel({ ...base, avg_tps: 41.6 })).toBe('42 t/s')
    expect(timeToFirstTokenLabel({ ...base, avg_ttft_s: 1.84 })).toBe('1.8s')
    expect(timeToFirstTokenLabel({ ...base, avg_ttft_s: 0.42 })).toBe('420ms')
  })
})
