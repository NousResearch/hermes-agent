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
  it('records a preview refusal as the previewFailed phase and re-runs on retry', async () => {
    apiMocks.previewSubscriptionChange.mockResolvedValue({
      ok: false,
      refusal: { kind: 'insufficient_scope', message: 'billing:manage required' }
    })

    const { result } = renderHook(() => useDowngradeFlow({ onScheduled: vi.fn() }), { wrapper })

    act(() => result.current.begin({ tierId: 't_free', tierName: 'Free' }))

    await waitFor(() => expect(result.current.active?.phase.kind).toBe('previewFailed'))
    const phase = result.current.active?.phase

    if (phase?.kind !== 'previewFailed') {
      throw new Error('expected previewFailed phase')
    }

    expect(phase.refusal.kind).toBe('insufficient_scope')

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
    await waitFor(() => expect(result.current.active?.phase.kind).toBe('ready'))

    act(() => result.current.cancel())

    expect(result.current.active).toBeNull()
    expect(apiMocks.scheduleSubscriptionChange).not.toHaveBeenCalled()
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
    await waitFor(() => expect(result.current.active?.phase.kind).toBe('ready'))
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
    await waitFor(() => expect(result.current.active?.phase.kind).toBe('ready'))

    // Two synchronous activations before React commits busy='schedule'.
    await act(async () => {
      void result.current.confirm()
      void result.current.confirm()
    })

    expect(apiMocks.scheduleSubscriptionChange).toHaveBeenCalledTimes(1)
  })
})

describe('useResumeFlow', () => {
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
})
