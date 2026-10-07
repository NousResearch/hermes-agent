import { describe, expect, it } from 'vitest'

import {
  cacheHitLabel,
  streamTpsLabel,
  tokenCountersLabel,
  tokensPerSecondLabel
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

  it('streamTpsLabel paints the live streaming rate and stays blank when absent or 0', () => {
    // The reducer pushes `stream_tps` only mid-turn — before the first delta,
    // and again after `message.complete`, it is absent/undefined, so the readout
    // must self-mask rather than paint a stale number.
    expect(streamTpsLabel(base)).toBe('')
    expect(streamTpsLabel({ ...base, stream_tps: undefined })).toBe('')
    expect(streamTpsLabel({ ...base, stream_tps: 0 })).toBe('')

    // Rounded to a whole number, same `N t/s` shape as tokensPerSecondLabel.
    expect(streamTpsLabel({ ...base, stream_tps: 7 })).toBe('7 t/s')
    expect(streamTpsLabel({ ...base, stream_tps: 12.6 })).toBe('13 t/s')
  })

  it('tokenCountersLabel paints ↑input ↓output and stays blank when both are 0', () => {
    // `UsageStats.input/output` are required `number` fields defaulting to 0,
    // so the "absent" case lands as a 0/0 pair — nothing to paint.
    expect(tokenCountersLabel(base)).toBe('')

    // Aligns with the TUI's `↑ … ↓ …` arrows. Both sides are required: a single
    // side by itself is a half-open counter, not a delta worth showing.
    expect(tokenCountersLabel({ ...base, input: 120, output: 340 })).toBe('↑ 120 ↓ 340')
    expect(tokenCountersLabel({ ...base, input: 120, output: 0 })).toBe('')
    expect(tokenCountersLabel({ ...base, input: 0, output: 340 })).toBe('')
  })
})
