import type { BillingCardInfo, BillingStateResponse, KnownBillingRefusalCode } from './billing-types.js'

export type BillingRecovery = 'login' | 'none' | 'portal' | 'reconnect' | 'retry' | 'step_up'

export interface BillingRefusalPolicy {
  recovery: BillingRecovery
  ambiguousMidPoll?: true
  reuseIdempotencyKey?: true
}

export const BILLING_REFUSAL_POLICY: Record<KnownBillingRefusalCode, BillingRefusalPolicy> = {
  auto_top_up_disabled_failures: { recovery: 'portal' },
  cli_billing_disabled: { recovery: 'portal' },
  consent_required: { recovery: 'portal' },
  endpoint_unavailable: { recovery: 'retry', reuseIdempotencyKey: true },
  idempotency_conflict: { recovery: 'none' },
  idempotency_key_required: { recovery: 'none' },
  // Deliberate: losing scope mid-poll cannot undo an accepted charge, so its outcome is unknown.
  insufficient_scope: { recovery: 'step_up', ambiguousMidPoll: true },
  internal_error: { recovery: 'retry' },
  invalid_charge_id: { recovery: 'none' },
  invalid_request: { recovery: 'none' },
  monthly_cap_exceeded: { recovery: 'portal' },
  network_error: { recovery: 'retry', reuseIdempotencyKey: true },
  no_payment_method: { recovery: 'portal' },
  org_access_denied: { recovery: 'portal' },
  preview_rejected: { recovery: 'none' },
  rate_limited: { recovery: 'retry', reuseIdempotencyKey: true },
  remote_spending_disabled: { recovery: 'portal' },
  remote_spending_revoked: { recovery: 'reconnect', ambiguousMidPoll: true },
  role_required: { recovery: 'portal' },
  session_revoked: { recovery: 'login', ambiguousMidPoll: true },
  stripe_unavailable: { recovery: 'retry', reuseIdempotencyKey: true },
  temporarily_unavailable: { recovery: 'retry', reuseIdempotencyKey: true },
  upgrade_cap_exceeded: { recovery: 'none' },
  validation_failed: { recovery: 'none' }
}

/**
 * What a credit purchase is paid with. `managed_payments`: Link bills it and adds
 * any sales tax or VAT on top, so the amount shown is before tax. `card`: the card
 * on file is charged the amount shown.
 */
export type PurchasePayer =
  | { card: BillingCardInfo; kind: 'card'; label: string }
  | { kind: 'managed_payments'; label: string }

/**
 * The payer a credit purchase uses, or null when nothing can pay yet (the add-card
 * path comes first). A Managed Payments payer wins over the card on file; auto-reload
 * keeps reading `card`, since only purchases go through Link.
 */
export function purchasePayer(state: Pick<BillingStateResponse, 'card' | 'managed_payments'>): null | PurchasePayer {
  if (state.managed_payments) {
    return { kind: 'managed_payments', label: state.managed_payments.label }
  }

  return state.card ? { card: state.card, kind: 'card', label: state.card.display ?? state.card.masked } : null
}

export function refusalPolicy(code: string): BillingRefusalPolicy {
  if (Object.hasOwn(BILLING_REFUSAL_POLICY, code)) {
    return BILLING_REFUSAL_POLICY[code as KnownBillingRefusalCode]
  }

  // Unknown codes must still show the server message; no special handling is the safe fallback.
  return { recovery: 'none' }
}
