import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import type { ContextBreakdown, UsageStats } from '@/types/hermes'

import { ContextUsagePanel } from './context-usage-panel'

afterEach(cleanup)

const usage = (contextUsed: number): UsageStats => ({
  calls: 1,
  input: contextUsed,
  output: 0,
  total: contextUsed,
  context_used: contextUsed,
  context_max: 372_000,
  context_percent: (contextUsed / 372_000) * 100
})

const breakdown = (contextUsed: number): ContextBreakdown => ({
  categories: [],
  context_max: 372_000,
  context_percent: (contextUsed / 372_000) * 100,
  context_used: contextUsed,
  estimated_total: contextUsed
})

describe('ContextUsagePanel reconciliation', () => {
  it('refetches the authoritative breakdown when current usage changes', async () => {
    const updatedBreakdown = new Promise<ContextBreakdown>(() => undefined)
    const requestGateway = vi.fn().mockResolvedValueOnce(breakdown(100_000)).mockReturnValueOnce(updatedBreakdown)

    const view = render(
      <I18nProvider configClient={null} initialLocale="en">
        <ContextUsagePanel currentUsage={usage(100_000)} requestGateway={requestGateway} sessionId="runtime-1" />
      </I18nProvider>
    )

    await waitFor(() => expect(requestGateway).toHaveBeenCalledTimes(1))

    view.rerender(
      <I18nProvider configClient={null} initialLocale="en">
        <ContextUsagePanel currentUsage={usage(200_000)} requestGateway={requestGateway} sessionId="runtime-1" />
      </I18nProvider>
    )

    await waitFor(() => expect(requestGateway).toHaveBeenCalledTimes(2))
    expect(screen.getByText(/~200k \/ 372k/)).toBeTruthy()
    expect(requestGateway).toHaveBeenLastCalledWith('session.context_breakdown', { session_id: 'runtime-1' })
  })
})
