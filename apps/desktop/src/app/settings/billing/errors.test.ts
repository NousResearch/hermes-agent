import type { KnownBillingRefusalCode } from '@hermes/shared/billing'
import { describe, expect, it } from 'vitest'

import type { BillingRefusal } from './api'
import { resolveRefusal } from './errors'

const expectedActions: Record<
  KnownBillingRefusalCode | 'timeout' | 'transport',
  'none' | 'portal' | 'retry' | 'step_up'
> = {
  auto_top_up_disabled_failures: 'none',
  cli_billing_disabled: 'portal',
  consent_required: 'portal',
  endpoint_unavailable: 'retry',
  idempotency_conflict: 'none',
  idempotency_key_required: 'none',
  insufficient_scope: 'step_up',
  internal_error: 'none',
  invalid_charge_id: 'none',
  invalid_request: 'none',
  monthly_cap_exceeded: 'portal',
  network_error: 'none',
  no_payment_method: 'portal',
  org_access_denied: 'none',
  preview_rejected: 'none',
  rate_limited: 'retry',
  remote_spending_disabled: 'portal',
  remote_spending_revoked: 'portal',
  role_required: 'portal',
  session_revoked: 'portal',
  stripe_unavailable: 'retry',
  temporarily_unavailable: 'retry',
  timeout: 'retry',
  transport: 'retry',
  upgrade_cap_exceeded: 'none',
  validation_failed: 'none'
}

describe('resolveRefusal', () => {
  it('maps every known refusal kind to copy and the expected action', () => {
    for (const [kind, actionType] of Object.entries(expectedActions)) {
      const resolved = resolveRefusal({
        kind: kind as BillingRefusal['kind'],
        message: 'Server message.',
        portalUrl: 'https://portal.nousresearch.com/billing',
        retryAfter: 90
      })

      expect(resolved.title, kind).not.toHaveLength(0)
      expect(resolved.message, kind).not.toHaveLength(0)
      expect(resolved.action.type, kind).toBe(actionType)
    }
  })

  it('includes monthly cap headroom when the server sends it', () => {
    const resolved = resolveRefusal({
      kind: 'monthly_cap_exceeded',
      message: 'Monthly spend cap reached.',
      payload: { remainingUsd: '4.50' }
    })

    expect(resolved.message).toContain('$4.50 headroom left')
  })

  it('includes Stripe retry timing when the server sends it', () => {
    const resolved = resolveRefusal({
      kind: 'stripe_unavailable',
      message: 'Stripe is unavailable.',
      retryAfter: 120
    })

    expect(resolved.message).toContain('try again in ~2 min')
  })

  it('falls back sanely for unknown refusal kinds', () => {
    const resolved = resolveRefusal({ kind: 'new_billing_code', message: 'Something changed upstream.' })

    expect(resolved).toEqual({
      action: { type: 'none' },
      message: 'Something changed upstream.',
      title: 'Billing request failed'
    })
  })
})
