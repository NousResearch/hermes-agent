import { describe, expect, it } from 'vitest'

import { cacheHitClass, cacheHitLabel, compressionCountClass, contextUsageClass, latencyLabel, tokensPerSecondLabel } from '@/lib/statusbar'

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

  it('formats rolling API latency with one decimal (CLI ◷ parity)', () => {
    expect(latencyLabel(base)).toBe('')
    expect(latencyLabel({ ...base, avg_latency_s: 0 })).toBe('')
    expect(latencyLabel({ ...base, avg_latency_s: 11.94 })).toBe('11.9s')
  })

  it('ladders context usage through the CLI thresholds (95/80/50)', () => {
    expect(contextUsageClass({ ...base, context_percent: 34 })).toBe('')
    expect(contextUsageClass({ ...base, context_percent: 50 })).toContain('text-(--ui-yellow)')
    expect(contextUsageClass({ ...base, context_percent: 81 })).toContain('text-(--ui-orange)')
    expect(contextUsageClass({ ...base, context_percent: 95 })).toContain('text-destructive')
    // No reading yet — unstyled, never alarm-colored.
    expect(contextUsageClass(base)).toBe('')
  })

  it('ladders cache hit rate INVERTED — a low hit rate is the expensive state (70/40)', () => {
    expect(cacheHitClass({ ...base, cache_hit_pct: 88.6 })).toContain('text-(--ui-green)')
    expect(cacheHitClass({ ...base, cache_hit_pct: 45 })).toContain('text-(--ui-yellow)')
    expect(cacheHitClass({ ...base, cache_hit_pct: 39 })).toContain('text-(--ui-orange)')
    expect(cacheHitClass(base)).toBe('')
  })

  it('ladders compression counts (≥10 destructive, ≥5 caution, else quiet)', () => {
    expect(compressionCountClass(1)).toBe('')
    expect(compressionCountClass(5)).toContain('text-(--ui-yellow)')
    expect(compressionCountClass(10)).toContain('text-destructive')
  })
})
