import { describe, expect, it } from 'vitest'

import { COMMAND_CENTER_ROUTE } from '@/app/routes'
import type { UsageMonthProvider, UsageMonthResponse } from '@/types/hermes'

import { usageBudgetChip } from './usage-meter-items'

const provider = (name: string, usedRatio: null | number): UsageMonthProvider => ({
  actual_cost: 0,
  budget:
    usedRatio === null
      ? null
      : {
          kind: 'tokens',
          limit: 100,
          projected_ratio: usedRatio,
          runs_out_on: null,
          used: usedRatio * 100,
          used_ratio: usedRatio
        },
  estimated_cost: 0,
  provider: name,
  sessions: 1,
  tokens: 10,
  unpriced_sessions: 0
})

const month = (providers: UsageMonthProvider[]): UsageMonthResponse => ({
  days_elapsed: 9.5,
  days_in_month: 31,
  month: '2026-10',
  providers
})

describe('usage budget status-bar chip', () => {
  it('names the budget nearest its limit and opens Command Center usage', () => {
    // The backend sorts budgeted providers nearest-limit first.
    const chip = usageBudgetChip(month([provider('xiaomi', 0.62), provider('openrouter', 0.1), provider('zai', null)]))

    expect(chip.label).toBe('xiaomi 62%')
    expect(chip.hidden).toBe(false)
    expect(chip.to).toBe(`${COMMAND_CENTER_ROUTE}?section=usage`)
  })

  it('stays out of the bar until a budget exists', () => {
    expect(usageBudgetChip(month([provider('zai', null)])).hidden).toBe(true)
    expect(usageBudgetChip(null).hidden).toBe(true)
  })
})
