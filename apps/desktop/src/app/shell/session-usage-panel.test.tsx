import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { UsageStats } from '@/types/hermes'

import { SessionUsagePanel } from './session-usage-panel'

const usage: UsageStats = {
  calls: 7,
  cache_hit_pct: 87,
  cache_read: 128_400,
  cache_write: 8_100,
  cost_status: 'estimated',
  cost_usd: 0.0312,
  input: 148_000,
  output: 21_000,
  total: 169_000
}

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe('SessionUsagePanel', () => {
  it('shows the session totals, cache counts and estimated spend', () => {
    render(<SessionUsagePanel usage={usage} />)

    expect(screen.getByText('Prompt tokens')).toBeTruthy()
    expect(screen.getByText('148k')).toBeTruthy()
    expect(screen.getByText('21k')).toBeTruthy()
    expect(screen.getByText('128.4k')).toBeTruthy()
    expect(screen.getByText('87%')).toBeTruthy()
    expect(screen.getByText('~¥0.225')).toBeTruthy()
    expect(screen.getByText('≈ $0.0312 USD')).toBeTruthy()
    expect(screen.getByText(/Estimated from published rates/)).toBeTruthy()
  })

  it('never fabricates a price, a cache figure or a hit rate the backend omitted', () => {
    const bare: UsageStats = { calls: 0, input: 0, output: 0, total: 0 }
    render(<SessionUsagePanel usage={bare} />)

    // No cost_usd → the "no price" label, not ¥0.0000; no cache data → em dashes.
    expect(screen.getByText('No price')).toBeTruthy()
    expect(screen.queryByText('¥0.0000')).toBeNull()
    expect(screen.queryByText(/Estimated from published rates/)).toBeNull()
    expect(screen.getAllByText('—').length).toBe(3)
  })

  it('labels a subscription session as included instead of pricing it', () => {
    render(<SessionUsagePanel usage={{ ...usage, cost_status: 'included' }} />)

    expect(screen.getByText('Included in plan')).toBeTruthy()
  })
})
