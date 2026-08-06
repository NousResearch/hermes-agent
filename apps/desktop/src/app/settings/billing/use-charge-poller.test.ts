import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, renderHook } from '@testing-library/react'
import { createElement, type PropsWithChildren } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { BillingResult } from './api'
import type { BillingChargeStatusResponse } from './types'

const apiMocks = vi.hoisted(() => ({
  charge: vi.fn(),
  chargeStatus: vi.fn()
}))

vi.mock('./api', () => ({
  useBillingApi: () => ({
    charge: apiMocks.charge,
    chargeStatus: apiMocks.chargeStatus
  })
}))

import { CHARGE_POLL_CAP_MS, pollChargeSettlement, useChargeFlow } from './use-charge-poller'

const status = (overrides: Partial<BillingChargeStatusResponse> = {}): BillingResult<BillingChargeStatusResponse> => ({
  data: {
    ok: true,
    status: 'pending',
    ...overrides
  },
  ok: true
})

const refusal = (
  kind: string,
  overrides: Partial<Extract<BillingResult<BillingChargeStatusResponse>, { ok: false }>['refusal']> = {}
): BillingResult<BillingChargeStatusResponse> => ({
  ok: false,
  refusal: {
    kind,
    message: kind,
    ...overrides
  }
})

function controlledClock() {
  let current = 0
  const waits: number[] = []

  return {
    now: () => current,
    sleep: vi.fn(async (ms: number) => {
      waits.push(ms)
      current += ms
    }),
    waits
  }
}

function wrapper({ children }: PropsWithChildren) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  return createElement(QueryClientProvider, { client }, children)
}

beforeEach(() => {
  apiMocks.charge.mockReset()
  apiMocks.chargeStatus.mockReset()
})

afterEach(() => {
  vi.clearAllMocks()
})

describe('pollChargeSettlement', () => {
  it('settles after pending polls', async () => {
    const clock = controlledClock()

    const api = {
      chargeStatus: vi
        .fn()
        .mockResolvedValueOnce(status())
        .mockResolvedValueOnce(status())
        .mockResolvedValueOnce(status({ amount_usd: '25', status: 'settled' }))
    }

    const outcome = await pollChargeSettlement(api, 'ch_123', clock)

    expect(outcome).toMatchObject({ amountUsd: '25', kind: 'success' })
    expect(api.chargeStatus).toHaveBeenCalledTimes(3)
    expect(clock.waits).toEqual([2000, 2000])
  })

  it('returns a failed outcome with the charge failure reason', async () => {
    const clock = controlledClock()

    const api = {
      chargeStatus: vi.fn().mockResolvedValue(status({ reason: 'card_declined', status: 'failed' }))
    }

    const outcome = await pollChargeSettlement(api, 'ch_123', clock)

    expect(outcome).toMatchObject({
      kind: 'failure',
      message: 'Your card was declined. Try another card on the portal.',
      title: 'Charge failed'
    })
    expect(clock.waits).toEqual([])
  })

  it('backs off on rate limits, honors retryAfter, and keeps polling', async () => {
    const clock = controlledClock()

    const api = {
      chargeStatus: vi
        .fn()
        .mockResolvedValueOnce(refusal('rate_limited', { retryAfter: 7 }))
        .mockResolvedValueOnce(status({ amount_usd: '50', status: 'settled' }))
    }

    const outcome = await pollChargeSettlement(api, 'ch_123', clock)

    expect(outcome).toMatchObject({ amountUsd: '50', kind: 'success' })
    expect(clock.waits).toEqual([7000])
  })

  it('backs off when Stripe is unavailable and keeps polling', async () => {
    const clock = controlledClock()

    const api = {
      chargeStatus: vi
        .fn()
        .mockResolvedValueOnce(refusal('stripe_unavailable', { retryAfter: 3 }))
        .mockResolvedValueOnce(status({ amount_usd: '50', status: 'settled' }))
    }

    const outcome = await pollChargeSettlement(api, 'ch_123', clock)

    expect(outcome).toMatchObject({ amountUsd: '50', kind: 'success' })
    expect(clock.waits).toEqual([3000])
  })

  it('caps pending polling at 5 minutes as an ambiguous outcome', async () => {
    const clock = controlledClock()

    const api = {
      chargeStatus: vi.fn().mockResolvedValue(status())
    }

    const outcome = await pollChargeSettlement(api, 'ch_123', {
      ...clock,
      portalUrl: 'https://portal.nousresearch.com/billing'
    })

    expect(outcome).toEqual({
      kind: 'ambiguous',
      message: 'Charge may still settle. Check the portal before retrying.',
      portalUrl: 'https://portal.nousresearch.com/billing',
      title: 'Still processing after 5 minutes'
    })
    expect(clock.waits.reduce((total, ms) => total + ms, 0)).toBe(CHARGE_POLL_CAP_MS)
  })

  it('treats auth revocation while polling as ambiguous', async () => {
    const clock = controlledClock()

    const api = {
      chargeStatus: vi.fn().mockResolvedValue(
        refusal('session_revoked', {
          message: 'Your session was logged out.',
          portalUrl: 'https://portal.nousresearch.com/billing'
        })
      )
    }

    const outcome = await pollChargeSettlement(api, 'ch_123', clock)

    expect(outcome).toMatchObject({
      kind: 'ambiguous',
      portalUrl: 'https://portal.nousresearch.com/billing',
      title: 'Charge outcome unconfirmed'
    })
  })
})

describe('useChargeFlow', () => {
  it('turns a charge refusal into an immediate outcome without polling', async () => {
    apiMocks.charge.mockResolvedValue({
      idempotencyKey: 'key-1',
      ok: false,
      refusal: {
        kind: 'no_payment_method',
        message: 'No saved card.',
        portalUrl: 'https://portal.nousresearch.com/billing'
      }
    })

    const { result } = renderHook(() => useChargeFlow(), { wrapper })

    await act(async () => {
      await result.current.start('25')
    })

    expect(result.current.phase).toBe('done')
    expect(result.current.outcome).toMatchObject({
      kind: 'failure',
      title: 'No saved card'
    })
    expect(apiMocks.chargeStatus).not.toHaveBeenCalled()
  })

  it('reuses one idempotency key when retrying a failed-to-send charge', async () => {
    apiMocks.charge.mockResolvedValue({
      idempotencyKey: 'key-1',
      ok: false,
      refusal: {
        kind: 'transport',
        message: 'connection closed'
      }
    })

    const { result } = renderHook(() => useChargeFlow(), { wrapper })

    await act(async () => {
      await result.current.start('25')
    })
    await act(async () => {
      await result.current.start('25')
    })

    expect(apiMocks.charge).toHaveBeenNthCalledWith(1, '25', undefined)
    expect(apiMocks.charge).toHaveBeenNthCalledWith(2, '25', 'key-1')
  })
})
