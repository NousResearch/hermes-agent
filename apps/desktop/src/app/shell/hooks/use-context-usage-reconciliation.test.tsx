import { cleanup, render, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { UsageStats } from '@/types/hermes'

import {
  reconcileFocusedContextUsage,
  useContextUsageReconciliation
} from './use-context-usage-reconciliation'

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

function Harness({
  gatewayState,
  onUsage,
  requestGateway,
  sessionId
}: {
  gatewayState: string
  onUsage: (next: UsageStats) => void
  requestGateway: <T = unknown>(method: string, params?: Record<string, unknown>) => Promise<T>
  sessionId: null | string
}) {
  useContextUsageReconciliation({ gatewayState, onUsage, requestGateway, sessionId })

  return null
}

describe('useContextUsageReconciliation', () => {
  it('reads authoritative usage when the gateway becomes available', async () => {
    const onUsage = vi.fn()
    const requestGateway = vi.fn().mockResolvedValue(usage(200_000))

    const view = render(
      <Harness gatewayState="connecting" onUsage={onUsage} requestGateway={requestGateway} sessionId="runtime-1" />
    )

    expect(requestGateway).not.toHaveBeenCalled()

    view.rerender(
      <Harness gatewayState="open" onUsage={onUsage} requestGateway={requestGateway} sessionId="runtime-1" />
    )

    await waitFor(() => expect(onUsage).toHaveBeenCalledWith(usage(200_000)))
    expect(requestGateway).toHaveBeenCalledWith('session.usage', { session_id: 'runtime-1' })
  })

  it('ignores a stale response after the focused runtime session changes', async () => {
    let resolveOld!: (value: UsageStats) => void

    const oldResponse = new Promise<UsageStats>(resolve => {
      resolveOld = resolve
    })

    const onUsage = vi.fn()

    const requestGateway = vi
      .fn()
      .mockReturnValueOnce(oldResponse)
      .mockResolvedValueOnce(usage(250_000))

    const view = render(
      <Harness gatewayState="open" onUsage={onUsage} requestGateway={requestGateway} sessionId="runtime-old" />
    )

    view.rerender(
      <Harness gatewayState="open" onUsage={onUsage} requestGateway={requestGateway} sessionId="runtime-new" />
    )

    await waitFor(() => expect(onUsage).toHaveBeenCalledWith(usage(250_000)))

    resolveOld(usage(100_000))
    await Promise.resolve()

    expect(onUsage).toHaveBeenCalledTimes(1)
  })
})

describe('reconcileFocusedContextUsage', () => {
  it('updates both the runtime cache and primary usage when the primary session is focused', () => {
    const updateCachedUsage = vi.fn()
    const updatePrimaryUsage = vi.fn()
    const nextUsage = usage(200_000)

    reconcileFocusedContextUsage({
      primaryFocused: true,
      updateCachedUsage,
      updatePrimaryUsage,
      usage: nextUsage
    })

    expect(updateCachedUsage).toHaveBeenCalledWith(nextUsage)
    expect(updatePrimaryUsage).toHaveBeenCalledWith(nextUsage)
  })

  it('updates only the runtime cache when a secondary session is focused', () => {
    const updateCachedUsage = vi.fn()
    const updatePrimaryUsage = vi.fn()
    const nextUsage = usage(250_000)

    reconcileFocusedContextUsage({
      primaryFocused: false,
      updateCachedUsage,
      updatePrimaryUsage,
      usage: nextUsage
    })

    expect(updateCachedUsage).toHaveBeenCalledWith(nextUsage)
    expect(updatePrimaryUsage).not.toHaveBeenCalled()
  })
})
