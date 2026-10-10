import { afterEach, describe, expect, it, vi } from 'vitest'

const getUsageMonth = vi.fn()

vi.mock('@/hermes', () => ({ getUsageMonth: (...args: unknown[]) => getUsageMonth(...args) }))

import { $usageMonth, $usageMonthState, refreshUsageMonth } from './usage-month'

const month = (tokens: number) => ({
  days_elapsed: 9.5,
  days_in_month: 31,
  month: '2026-10',
  providers: [
    {
      actual_cost: 0,
      budget: null,
      estimated_cost: 0,
      provider: 'xiaomi',
      sessions: 1,
      tokens,
      unpriced_sessions: 1
    }
  ]
})

afterEach(() => {
  getUsageMonth.mockReset()
  $usageMonth.set(null)
})

describe('usage month store', () => {
  it('loads the month and clears the loading state', async () => {
    getUsageMonth.mockResolvedValue(month(5))
    const pending = refreshUsageMonth()

    expect($usageMonthState.get().loading).toBe(true)
    await pending
    expect($usageMonth.get()?.providers[0].tokens).toBe(5)
    expect($usageMonthState.get()).toMatchObject({ error: '', loading: false })
  })

  it('never lets an older response overwrite a newer one', async () => {
    let resolveOld: (value: unknown) => void = () => {}

    getUsageMonth
      .mockImplementationOnce(() => new Promise(resolve => (resolveOld = resolve)))
      .mockResolvedValueOnce(month(2))
    const old = refreshUsageMonth()

    await refreshUsageMonth()
    resolveOld(month(1))
    await old
    expect($usageMonth.get()?.providers[0].tokens).toBe(2)
  })

  it('keeps the last good month and reports the error', async () => {
    getUsageMonth.mockResolvedValueOnce(month(3)).mockRejectedValueOnce(new Error('offline'))
    await refreshUsageMonth()
    await refreshUsageMonth()
    expect($usageMonth.get()?.providers[0].tokens).toBe(3)
    expect($usageMonthState.get()).toMatchObject({ error: 'offline', loading: false })
  })
})
