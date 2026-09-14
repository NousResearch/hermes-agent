import { describe, expect, it } from 'vitest'

import {
  cacheHitLabel,
  contextBarLabel,
  sessionUsageTotalLabel,
  tokensPerSecondLabel,
  usageContextLabel
} from '@/lib/statusbar'

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

  it('labels the session LIFETIME total with a Σ, from total or the in/out split', () => {
    expect(sessionUsageTotalLabel(base)).toBe('')

    const usage = { ...base, input: 120_000, output: 62_400, total: 182_400 }
    expect(sessionUsageTotalLabel(usage)).toBe('Σ182.4k tok')
    // `total` absent (older/provider payloads): the split still answers.
    expect(sessionUsageTotalLabel({ ...usage, total: 0 })).toBe('Σ182.4k tok')
  })

  it('renders the context window and its meter, `~`-marked when the figure is estimated', () => {
    const measured = { ...base, context_max: 200_000, context_percent: 62, context_used: 124_000 }
    expect(usageContextLabel(measured)).toBe('124k/200k')
    expect(contextBarLabel(measured)).toBe('[██████░░░░] 62%')

    const estimated = { ...measured, context_estimated: true, context_percent: 62.4 }
    expect(usageContextLabel(estimated)).toBe('~124k/200k')
    expect(contextBarLabel(estimated)).toBe('[██████░░░░] ~62%')

    // No window known: fall back to the lifetime total; '' before any call.
    expect(usageContextLabel(base)).toBe('')
    expect(contextBarLabel(base)).toBe('')
    expect(usageContextLabel({ ...base, total: 182_400 })).toBe('182.4k tok')
  })
})
