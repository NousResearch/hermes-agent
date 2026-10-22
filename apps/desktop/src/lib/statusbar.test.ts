import { describe, expect, it } from 'vitest'

import { cacheHitLabel, costMerge, sessionCostLabel, tokensPerSecondLabel } from '@/lib/statusbar'

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

describe('session cost label', () => {
  it('marks an estimate and leaves a provider-reported figure unmarked', () => {
    expect(sessionCostLabel({ ...base, cost_status: 'estimated', cost_usd: 4.271 })).toBe('~$4.27')
    expect(sessionCostLabel({ ...base, cost_status: 'actual', cost_usd: 4.271 })).toBe('$4.27')
  })

  it('reports a subscription session as $0.00 — that is a measurement, not a guess', () => {
    // `included` means the calls were covered; the zero is definitive, so it
    // carries no `~`. The chip must be able to say this or a subscriber's
    // toggle stays blank forever.
    expect(sessionCostLabel({ ...base, cost_status: 'included', cost_usd: 0 })).toBe('$0.00')
  })

  it('stays blank when the backend sent no figure or an unusable one', () => {
    // The exact dishonesty `fd2a35b169` removed: a missing measurement painted
    // as $0.00 reads as "this was free" rather than "we do not know yet".
    expect(sessionCostLabel(base)).toBe('')
    expect(sessionCostLabel({ ...base, cost_status: 'unknown', cost_usd: 0 })).toBe('')
    expect(sessionCostLabel({ ...base, cost_status: 'estimated', cost_usd: Number.NaN })).toBe('')
    expect(sessionCostLabel({ ...base, cost_status: 'estimated', cost_usd: -1 })).toBe('')
  })

  it('keeps sub-cent spend readable instead of rounding it to a misleading $0.00', () => {
    // Same rule as `agent/usage_pricing.py::format_cost_label` (#79220).
    expect(sessionCostLabel({ ...base, cost_status: 'estimated', cost_usd: 0.0046 })).toBe('~$0.0046')
    expect(sessionCostLabel({ ...base, cost_status: 'estimated', cost_usd: 0.00001 })).toBe('~$<0.0001')
    expect(sessionCostLabel({ ...base, cost_status: 'actual', cost_usd: 0.0046 })).toBe('$0.0046')
  })
})

describe('cost merge', () => {
  it('refuses a figure smaller than the one on screen', () => {
    // The gateway rebuilds every counter at 0 on resume, so the first turn after
    // one reports only its own cost. Letting that through dropped a $5.00
    // session to $0.02 — the readout must never run backwards, so the merge
    // hands back the value to keep.
    const shown = { ...base, cost_status: 'estimated', cost_usd: 5 }

    expect(costMerge(shown, { cost_status: 'estimated', cost_usd: 0.02 })).toEqual({
      cost_status: 'estimated',
      cost_usd: 5
    })
  })

  it('adopts a larger figure with its status', () => {
    const shown = { ...base, cost_status: 'estimated', cost_usd: 5 }

    expect(costMerge(shown, { cost_status: 'actual', cost_usd: 5.5 })).toEqual({
      cost_status: 'actual',
      cost_usd: 5.5
    })
  })

  it('leaves the figure alone when the tick carries none', () => {
    expect(costMerge({ ...base, cost_usd: 5 }, {})).toEqual({})
  })
})
