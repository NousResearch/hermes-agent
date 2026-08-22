import { describe, expect, it } from 'vitest'

import type { ContextBreakdown, UsageStats } from '@/types/hermes'

import { resolveContextGaugeUsage } from './use-statusbar-items'

// Regression for #70871's frozen-gauge half: while a turn is streaming the
// pre-turn context breakdown must NOT overlay the live streamed usage —
// otherwise the bar stays frozen at the turn-start value and jumps at turn end.
const CURRENT: UsageStats = {
  calls: 1,
  context_max: 1_000_000,
  context_percent: 27,
  context_used: 267_700,
  input: 10_000,
  output: 5_000,
  total: 15_000
}

const BREAKDOWN: ContextBreakdown = {
  categories: [],
  context_max: 1_000_000,
  context_percent: 30,
  context_used: 320_000,
  estimated_total: 350_000,
  model: 'test-model'
}

describe('resolveContextGaugeUsage', () => {
  it('keeps the live streamed usage mid-turn even when a breakdown is held', () => {
    const out = resolveContextGaugeUsage({ busy: true, breakdown: BREAKDOWN, current: CURRENT })

    expect(out.context_used).toBe(267_700)
    expect(out.context_percent).toBe(27)
    expect(out.input).toBe(10_000)
  })

  it('falls back to the streamed usage mid-turn when no breakdown is held', () => {
    const out = resolveContextGaugeUsage({ busy: true, breakdown: null, current: CURRENT })

    expect(out).toBe(CURRENT)
  })

  it('lets the breakdown win when the turn is idle', () => {
    const out = resolveContextGaugeUsage({ busy: false, breakdown: BREAKDOWN, current: CURRENT })

    expect(out.context_used).toBe(320_000)
    expect(out.context_percent).toBe(30)
    expect(out.context_max).toBe(1_000_000)
    // Non-context fields still come from the streamed usage object.
    expect(out.input).toBe(10_000)
    expect(out.calls).toBe(1)
  })

  it('keeps the streamed usage when idle without a breakdown', () => {
    const out = resolveContextGaugeUsage({ busy: false, breakdown: undefined, current: CURRENT })

    expect(out).toBe(CURRENT)
  })
})
