import { compactNumber } from '@hermes/shared'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

const getUsageMonth = vi.fn()
const setUsageBudget = vi.fn()

vi.mock('@/hermes', () => ({
  getUsageMonth: (...args: unknown[]) => getUsageMonth(...args),
  setUsageBudget: (...args: unknown[]) => setUsageBudget(...args)
}))

import { enCommandCenter as cc } from '@/i18n/en_command_center'
import { $usageMonth } from '@/store/usage-month'
import type { UsageMonthProvider } from '@/types/hermes'

import { formatRunsOut, UsageMonthSection } from './usage-month'

const xiaomi: UsageMonthProvider = {
  actual_cost: 0,
  budget: null,
  estimated_cost: 0,
  provider: 'xiaomi',
  sessions: 306,
  tokens: 190_000_000,
  unpriced_sessions: 306
}

const month = (providers: UsageMonthProvider[]) => ({
  days_elapsed: 9.5,
  days_in_month: 31,
  month: '2026-10',
  providers
})

afterEach(() => {
  cleanup()
  getUsageMonth.mockReset()
  setUsageBudget.mockReset()
  $usageMonth.set(null)
})

describe('UsageMonthSection', () => {
  it('counts an unpriced provider in tokens and never prices it at $0', async () => {
    getUsageMonth.mockResolvedValue(month([xiaomi]))
    render(<UsageMonthSection />)

    await screen.findByText(cc.unpricedSessions(306))
    expect(screen.getByText(cc.tokensUsed(compactNumber(190_000_000)))).toBeTruthy()
    expect(screen.queryByText(/\$0/)).toBeNull()
  })

  it('shows the budget, the month-end pace and the overrun date', async () => {
    getUsageMonth.mockResolvedValue(
      month([
        {
          ...xiaomi,
          budget: {
            kind: 'tokens',
            limit: 500_000_000,
            projected_ratio: 1.24,
            runs_out_on: '2026-10-26',
            used: 190_000_000,
            used_ratio: 0.38
          }
        }
      ])
    )
    render(<UsageMonthSection />)

    await screen.findByText(cc.budgetOf(compactNumber(190_000_000), compactNumber(500_000_000)))
    expect(screen.getByText(cc.projected(124))).toBeTruthy()
    expect(screen.getByText(cc.runsOutOn(formatRunsOut('2026-10-26')))).toBeTruthy()
    expect(screen.getByRole('progressbar')).toBeTruthy()
  })

  it('sets a token budget and reloads the month', async () => {
    getUsageMonth.mockResolvedValue(month([xiaomi]))
    setUsageBudget.mockResolvedValue({ ok: true })
    render(<UsageMonthSection />)

    fireEvent.click(await screen.findByText(cc.setBudget))
    fireEvent.change(screen.getByRole('spinbutton'), { target: { value: '500000000' } })
    fireEvent.click(screen.getByText(cc.saveBudget))

    await waitFor(() =>
      expect(setUsageBudget).toHaveBeenCalledWith({ monthly_tokens: 500_000_000, provider: 'xiaomi' })
    )
    await waitFor(() => expect(getUsageMonth).toHaveBeenCalledTimes(2))
  })

  it('says what it is loading and for how long', () => {
    getUsageMonth.mockReturnValue(new Promise(() => {}))
    render(<UsageMonthSection />)

    expect(screen.getByText(cc.loadingMonth(0))).toBeTruthy()
  })
})
