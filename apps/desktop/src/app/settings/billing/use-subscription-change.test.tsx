import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, renderHook, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

const apiMocks = vi.hoisted(() => ({
  previewSubscriptionChange: vi.fn(),
  resumeSubscription: vi.fn(),
  scheduleSubscriptionChange: vi.fn()
}))

vi.mock('./api', () => ({ useBillingApi: () => apiMocks }))

import { useDowngradeFlow, useResumeFlow } from './use-subscription-change'

function wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  return <QueryClientProvider client={client}>{children}</QueryClientProvider>
}

afterEach(() => {
  vi.clearAllMocks()
  vi.unstubAllEnvs()
})

describe('useDowngradeFlow', () => {
  it('previews then schedules with the tier id, refetches, and calls onScheduled', async () => {
    apiMocks.previewSubscriptionChange.mockResolvedValue({
      data: { effect: 'scheduled', ok: true, target_tier_name: 'Free' },
      ok: true
    })
    apiMocks.scheduleSubscriptionChange.mockResolvedValue({ data: { ok: true }, ok: true })
    const onScheduled = vi.fn()

    const { result } = renderHook(() => useDowngradeFlow({ onScheduled }), { wrapper })

    act(() => result.current.begin({ tierId: 't_free', tierName: 'Free' }))

    await waitFor(() => expect(result.current.active?.preview?.effect).toBe('scheduled'))
    expect(apiMocks.previewSubscriptionChange).toHaveBeenCalledWith('t_free')

    await act(async () => {
      await result.current.confirm()
    })

    expect(apiMocks.scheduleSubscriptionChange).toHaveBeenCalledWith('t_free')
    expect(onScheduled).toHaveBeenCalledTimes(1)
    expect(result.current.active).toBeNull()
  })

  it('records a preview refusal as failedOp "preview" and re-runs on retry', async () => {
    apiMocks.previewSubscriptionChange.mockResolvedValue({
      ok: false,
      refusal: { kind: 'insufficient_scope', message: 'billing:manage required' }
    })

    const { result } = renderHook(() => useDowngradeFlow({ onScheduled: vi.fn() }), { wrapper })

    act(() => result.current.begin({ tierId: 't_free', tierName: 'Free' }))

    await waitFor(() => expect(result.current.active?.failedOp).toBe('preview'))
    expect(result.current.active?.refusal?.kind).toBe('insufficient_scope')

    act(() => result.current.retryPreview())

    await waitFor(() => expect(apiMocks.previewSubscriptionChange).toHaveBeenCalledTimes(2))
  })

  it('cancel clears the active change without scheduling', async () => {
    apiMocks.previewSubscriptionChange.mockResolvedValue({
      data: { effect: 'scheduled', ok: true, target_tier_name: 'Free' },
      ok: true
    })

    const { result } = renderHook(() => useDowngradeFlow({ onScheduled: vi.fn() }), { wrapper })

    act(() => result.current.begin({ tierId: 't_free', tierName: 'Free' }))
    await waitFor(() => expect(result.current.active?.preview).not.toBeNull())

    act(() => result.current.cancel())

    expect(result.current.active).toBeNull()
    expect(apiMocks.scheduleSubscriptionChange).not.toHaveBeenCalled()
  })

  it('simulates a canned scheduled preview without touching the gateway', async () => {
    const { result } = renderHook(
      () => useDowngradeFlow({ onScheduled: vi.fn(), simulate: { effectiveAt: '2026-08-15T00:00:00Z' } }),
      { wrapper }
    )

    act(() => result.current.begin({ tierId: 't_free', tierName: 'Plus' }))

    await waitFor(() => expect(result.current.active?.preview?.effect).toBe('scheduled'))
    expect(result.current.active?.preview?.target_tier_name).toBe('Plus')
    expect(result.current.active?.preview?.effective_at).toBe('2026-08-15T00:00:00Z')
    expect(apiMocks.previewSubscriptionChange).not.toHaveBeenCalled()
  })

  it('ignores the simulate seam outside DEV — production never takes the canned branch', async () => {
    vi.stubEnv('DEV', false)
    apiMocks.previewSubscriptionChange.mockResolvedValue({
      data: { effect: 'scheduled', ok: true, target_tier_name: 'Plus' },
      ok: true
    })

    const { result } = renderHook(
      () => useDowngradeFlow({ onScheduled: vi.fn(), simulate: { effectiveAt: '2026-08-15T00:00:00Z' } }),
      { wrapper }
    )

    act(() => result.current.begin({ tierId: 't_free', tierName: 'Plus' }))

    // Real RPC is called despite the simulate prop being set.
    await waitFor(() => expect(apiMocks.previewSubscriptionChange).toHaveBeenCalledWith('t_free'))
  })

  it('exposes mutating only while the schedule RPC is in flight', async () => {
    apiMocks.previewSubscriptionChange.mockResolvedValue({
      data: { effect: 'scheduled', ok: true, target_tier_name: 'Free' },
      ok: true
    })

    let settleSchedule: (value: unknown) => void = () => {}
    apiMocks.scheduleSubscriptionChange.mockReturnValue(
      new Promise(resolve => {
        settleSchedule = resolve
      })
    )

    const { result } = renderHook(() => useDowngradeFlow({ onScheduled: vi.fn() }), { wrapper })

    act(() => result.current.begin({ tierId: 't_free', tierName: 'Free' }))
    await waitFor(() => expect(result.current.active?.preview?.effect).toBe('scheduled'))
    expect(result.current.mutating).toBe(false)

    act(() => {
      void result.current.confirm()
    })
    await waitFor(() => expect(result.current.mutating).toBe(true))

    act(() => settleSchedule({ data: { ok: true }, ok: true }))
    await waitFor(() => expect(result.current.mutating).toBe(false))
  })

  it('fires a single schedule RPC when confirm is double-activated in the same tick', async () => {
    apiMocks.previewSubscriptionChange.mockResolvedValue({
      data: { effect: 'scheduled', ok: true, target_tier_name: 'Free' },
      ok: true
    })
    apiMocks.scheduleSubscriptionChange.mockResolvedValue({ data: { ok: true }, ok: true })

    const { result } = renderHook(() => useDowngradeFlow({ onScheduled: vi.fn() }), { wrapper })

    act(() => result.current.begin({ tierId: 't_free', tierName: 'Free' }))
    await waitFor(() => expect(result.current.active?.preview?.effect).toBe('scheduled'))

    // Two synchronous activations before React commits busy='schedule'.
    await act(async () => {
      void result.current.confirm()
      void result.current.confirm()
    })

    expect(apiMocks.scheduleSubscriptionChange).toHaveBeenCalledTimes(1)
  })
})

describe('useResumeFlow', () => {
  it('resumes (undo) and clears the refusal on success', async () => {
    apiMocks.resumeSubscription.mockResolvedValue({ data: { ok: true }, ok: true })

    const { result } = renderHook(() => useResumeFlow(), { wrapper })

    await act(async () => {
      await result.current.resume()
    })

    expect(apiMocks.resumeSubscription).toHaveBeenCalledTimes(1)
    expect(result.current.refusal).toBeNull()
  })

  it('surfaces a resume refusal', async () => {
    apiMocks.resumeSubscription.mockResolvedValue({
      ok: false,
      refusal: { kind: 'insufficient_scope', message: 'billing:manage required' }
    })

    const { result } = renderHook(() => useResumeFlow(), { wrapper })

    await act(async () => {
      await result.current.resume()
    })

    expect(result.current.refusal?.kind).toBe('insufficient_scope')
  })

  it('does not touch the gateway in simulate mode', async () => {
    const { result } = renderHook(() => useResumeFlow(true), { wrapper })

    await act(async () => {
      await result.current.resume()
    })

    expect(apiMocks.resumeSubscription).not.toHaveBeenCalled()
  })
})
