import { describe, expect, it } from 'vitest'

import { cacheHitLabel, contextBarLabel, tokensPerSecondLabel, usageContextLabel } from '@/lib/statusbar'

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

  it('reads an occupancy the owner has not measured as `—`, never as a zero', () => {
    // A committed model switch adopts the new window before anything is counted in it, so the
    // backend sends `context_max` and omits the occupancy (`gateway/session_usage.py::tui_usage`).
    const switched = { ...base, context_max: 500_000 }

    expect(usageContextLabel(switched)).toBe('—/500k')
    expect(contextBarLabel(switched)).toBe('')
    // A measured occupancy keeps painting the used/window pair and the bar.
    expect(usageContextLabel({ ...switched, context_percent: 8, context_used: 40_000 })).toBe('40k/500k')
    expect(contextBarLabel({ ...switched, context_percent: 8, context_used: 40_000 })).toBe('[█░░░░░░░░░] 8%')
  })
})
