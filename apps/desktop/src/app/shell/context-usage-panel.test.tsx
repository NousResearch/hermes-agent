import { cleanup, render, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { ContextBreakdown, UsageStats } from '@/types/hermes'

import { ContextUsagePanel } from './context-usage-panel'

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('ContextUsagePanel', () => {
  it('publishes the live breakdown usage snapshot so the status bar can re-baseline', async () => {
    const currentUsage: UsageStats = {
      calls: 1,
      input: 0,
      output: 0,
      total: 0,
      context_max: 272_000,
      context_used: 128_200,
      context_percent: 47
    }

    const breakdown: ContextBreakdown = {
      categories: [{ id: 'conversation', label: 'Conversation', color: 'teal', tokens: 241_400 }],
      context_max: 272_000,
      context_percent: 89,
      context_used: 241_400,
      estimated_total: 286_600,
      model: 'test-model'
    }

    const requestGateway = vi.fn().mockResolvedValue(breakdown)
    const onUsageSnapshot = vi.fn()

    render(
      <ContextUsagePanel
        currentUsage={currentUsage}
        onUsageSnapshot={onUsageSnapshot}
        requestGateway={requestGateway}
        sessionId="runtime-1"
      />
    )

    await waitFor(() => {
      expect(onUsageSnapshot).toHaveBeenCalledWith({
        context_max: 272_000,
        context_percent: 89,
        context_used: 241_400
      })
    })
  })
})
