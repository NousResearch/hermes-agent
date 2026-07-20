import { renderHook } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { BillingChargeResponse, BillingStateResponse } from './types'

const requestGatewayMock = vi.hoisted(() => vi.fn())

vi.mock('@/app/gateway/hooks/use-gateway-request', () => ({
  useGatewayRequest: () => ({ requestGateway: requestGatewayMock })
}))

import { createBillingApi, useBillingApi } from './api'

describe('createBillingApi', () => {
  beforeEach(() => {
    requestGatewayMock.mockReset()
    vi.restoreAllMocks()
  })

  it('passes successful RPC results through as data', async () => {
    const state = {
      auto_reload: null,
      balance_display: '$10.00',
      balance_usd: '10',
      can_charge: true,
      card: null,
      charge_presets: ['10'],
      charge_presets_display: ['$10'],
      cli_billing_enabled: true,
      is_admin: true,
      logged_in: true,
      max_usd: '100',
      min_usd: '10',
      monthly_cap: null,
      ok: true,
      org_name: 'Nous',
      portal_url: 'https://portal.nousresearch.com/billing',
      role: 'OWNER'
    } satisfies BillingStateResponse

    requestGatewayMock.mockResolvedValueOnce(state)

    const { result } = renderHook(() => useBillingApi())
    const response = await result.current.fetchBillingState()

    expect(response).toEqual({ data: state, ok: true })
    expect(requestGatewayMock).toHaveBeenCalledWith('billing.state', {})
  })

  it('normalizes object-shaped refusal envelopes', async () => {
    requestGatewayMock.mockResolvedValueOnce({
      error: {
        kind: 'no_payment_method',
        message: 'No saved card.',
        portal_url: 'https://portal.nousresearch.com/billing',
        retry_after: 30
      },
      ok: false
    })

    const api = createBillingApi(requestGatewayMock)
    const response = await api.chargeStatus('ch_123')

    expect(response).toMatchObject({
      ok: false,
      refusal: {
        kind: 'no_payment_method',
        message: 'No saved card.',
        portalUrl: 'https://portal.nousresearch.com/billing',
        retryAfter: 30
      }
    })
    expect(requestGatewayMock).toHaveBeenCalledWith('billing.charge_status', { charge_id: 'ch_123' })
  })

  it('normalizes current string-shaped refusal envelopes', async () => {
    requestGatewayMock.mockResolvedValueOnce({
      error: 'monthly_cap_exceeded',
      message: 'Monthly spend cap reached.',
      ok: false,
      payload: { remainingUsd: '4.50' },
      portal_url: 'https://portal.nousresearch.com/billing'
    })

    const api = createBillingApi(requestGatewayMock)
    const response = await api.updateAutoReload({ enabled: true, reload_to_usd: '100', threshold_usd: '25' })

    expect(response).toMatchObject({
      ok: false,
      refusal: {
        kind: 'monthly_cap_exceeded',
        message: 'Monthly spend cap reached.',
        payload: { remainingUsd: '4.50' },
        portalUrl: 'https://portal.nousresearch.com/billing'
      }
    })
    expect(requestGatewayMock).toHaveBeenCalledWith('billing.auto_reload', {
      enabled: true,
      threshold: '25',
      top_up_amount: '100'
    })
  })

  it('maps thrown gateway failures to transport refusals', async () => {
    requestGatewayMock.mockRejectedValueOnce(new Error('connection closed'))

    const api = createBillingApi(requestGatewayMock)
    const response = await api.fetchSubscriptionState()

    expect(response).toEqual({
      ok: false,
      refusal: {
        kind: 'transport',
        message: 'connection closed',
        raw: expect.any(Error)
      }
    })
  })

  it('maps thrown timeout failures to timeout refusals', async () => {
    requestGatewayMock.mockRejectedValueOnce(new Error('request timed out after 5000ms'))

    const api = createBillingApi(requestGatewayMock)
    const response = await api.stepUp()

    expect(response).toEqual({
      ok: false,
      refusal: {
        kind: 'timeout',
        message: 'request timed out after 5000ms',
        raw: expect.any(Error)
      }
    })
  })

  it('sends a step-up session id when provided', async () => {
    requestGatewayMock.mockResolvedValueOnce({ granted: true, ok: true })

    const api = createBillingApi(requestGatewayMock)
    await api.stepUp('session-123')

    expect(requestGatewayMock).toHaveBeenCalledWith('billing.step_up', { session_id: 'session-123' })
  })

  it('sends a minted charge idempotency key and reuses it on explicit retry', async () => {
    vi.spyOn(crypto, 'randomUUID').mockReturnValue('11111111-1111-4111-8111-111111111111')

    const submitted = {
      charge_id: 'ch_123',
      idempotency_key: '11111111-1111-4111-8111-111111111111',
      ok: true
    } satisfies BillingChargeResponse

    requestGatewayMock.mockResolvedValue(submitted)

    const api = createBillingApi(requestGatewayMock)
    const first = await api.charge('25')
    const second = await api.charge('25', first.idempotencyKey)

    expect(first).toEqual({ data: submitted, idempotencyKey: '11111111-1111-4111-8111-111111111111', ok: true })
    expect(second).toEqual({ data: submitted, idempotencyKey: '11111111-1111-4111-8111-111111111111', ok: true })
    expect(crypto.randomUUID).toHaveBeenCalledTimes(1)
    expect(requestGatewayMock).toHaveBeenNthCalledWith(1, 'billing.charge', {
      amount_usd: '25',
      idempotency_key: '11111111-1111-4111-8111-111111111111'
    })
    expect(requestGatewayMock).toHaveBeenNthCalledWith(2, 'billing.charge', {
      amount_usd: '25',
      idempotency_key: '11111111-1111-4111-8111-111111111111'
    })
  })
})
