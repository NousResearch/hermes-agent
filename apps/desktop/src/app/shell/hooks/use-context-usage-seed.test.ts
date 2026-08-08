import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { ContextBreakdown } from '@/types/hermes'

import { useContextUsageSeed } from './use-context-usage-seed'

const breakdown: ContextBreakdown = {
  categories: [{ color: 'teal', id: 'conversation', label: 'Conversation', tokens: 241_400 }],
  context_max: 272_000,
  context_percent: 89,
  context_used: 241_400,
  estimated_total: 286_600,
  model: 'test-model'
}

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe('useContextUsageSeed', () => {
  it('fetches the breakdown once and seeds context_max for the active session', async () => {
    const requestGateway = vi.fn().mockResolvedValue(breakdown)
    const published = vi.fn()

    const { result } = renderHook(() =>
      useContextUsageSeed({
        activeSessionId: 'runtime-1',
        contextMax: undefined,
        publishContextUsage: published,
        requestGateway
      })
    )

    await waitFor(() => {
      expect(published).toHaveBeenCalledWith({
        context_max: 272_000,
        context_percent: 89,
        context_used: 241_400
      })
      expect(requestGateway).toHaveBeenCalledWith('session.context_breakdown', { session_id: 'runtime-1' })
    })

    await act(async () => {})
    expect(requestGateway).toHaveBeenCalledTimes(1)
    expect(result.current).toBeUndefined()
  })

  it('does not fetch when context_max is already known', () => {
    const requestGateway = vi.fn()

    renderHook(() =>
      useContextUsageSeed({
        activeSessionId: 'runtime-1',
        contextMax: 272_000,
        publishContextUsage: vi.fn(),
        requestGateway
      })
    )

    expect(requestGateway).not.toHaveBeenCalled()
  })

  it('does not fetch when there is no active session', () => {
    const requestGateway = vi.fn()

    renderHook(() =>
      useContextUsageSeed({
        activeSessionId: null,
        contextMax: undefined,
        publishContextUsage: vi.fn(),
        requestGateway
      })
    )

    expect(requestGateway).not.toHaveBeenCalled()
  })

  it('does not seed when the fetch fails', async () => {
    const requestGateway = vi.fn().mockRejectedValue(new Error('backend unreachable'))
    const published = vi.fn()

    renderHook(() =>
      useContextUsageSeed({
        activeSessionId: 'runtime-1',
        contextMax: undefined,
        publishContextUsage: published,
        requestGateway
      })
    )

    await act(async () => {})
    expect(published).not.toHaveBeenCalled()
  })

  it('skips publishing a stale result after the session changes mid-fetch', async () => {
    const resolvers: Array<(data: ContextBreakdown) => void> = []

    const requestGateway = vi
      .fn()
      .mockImplementation(() => new Promise<ContextBreakdown>(r => resolvers.push(r)))

    const published = vi.fn()

    const { rerender } = renderHook(
      (props: { activeSessionId: string | null }) =>
        useContextUsageSeed({
          activeSessionId: props.activeSessionId,
          contextMax: undefined,
          publishContextUsage: published,
          requestGateway
        }),
      { initialProps: { activeSessionId: 'runtime-1' } }
    )

    // Switch sessions while the first fetch is still in flight; the first
    // effect's cleanup marks it cancelled.
    rerender({ activeSessionId: 'runtime-2' })

    await act(async () => {
      // Resolve only the FIRST (now-cancelled) fetch. It must not publish.
      resolvers[0]?.(breakdown)
    })

    expect(published).not.toHaveBeenCalled()
  })
})
