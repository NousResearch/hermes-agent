/**
 * Client-side billing types.
 *
 * The wire shapes themselves (usage bars, billing/subscription states, the
 * charge / step-up / preview / upgrade envelopes) are the generated contract in
 * `gateway-contract.generated.ts` — import those, do not mirror them here. This
 * module keeps only what the contract deliberately leaves open: the closed code
 * sets the client classifies against, and the error-payload fields it reads out
 * of the opaque `payload` JSON.
 */

// ── Billing wall (inference credit exhaustion) ───────────────────────

/**
 * Structured billing-wall descriptor emitted by the gateway on the
 * `message.complete` event (`payload.billing`) when an inference call fails
 * because the account is out of credits / payment is required — mirrors the
 * Python `agent/billing_links.py::BillingBlock`.
 *
 * Detection is backend-only (`agent/error_classifier.py` →
 * `FailoverReason.billing`), so every surface renders from this one signal and
 * never re-classifies free-form error text. `is_nous` routes recovery: Nous is
 * the managed route with in-app billing (desktop Settings → Billing, TUI
 * `/topup`), while third-party providers deep-link to `billing_url`.
 */
export type { BillingBlock } from './gateway-contract.generated.js'

// ── Remote Spending (Phase 2b) ───────────────────────────────────────

/**
 * The closed set of refusal/error codes the gateway serializes today
 * (`_serialize_billing_error` preserves the raw NAS code where one exists,
 * plus the client-originated transport codes). Closed on purpose: an
 * exhaustive `Record<KnownBillingRefusalCode, …>` (classification tables,
 * copy maps, tests) gets a compile error when a code is added here but not
 * mapped.
 */
export type KnownBillingRefusalCode =
  | 'auto_top_up_disabled_failures'
  | 'cli_billing_disabled'
  | 'consent_required'
  | 'endpoint_unavailable'
  | 'idempotency_conflict'
  | 'idempotency_key_required'
  | 'insufficient_scope'
  | 'internal_error'
  | 'invalid_charge_id'
  | 'invalid_request'
  | 'monthly_cap_exceeded'
  | 'network_error'
  | 'no_payment_method'
  | 'org_access_denied'
  | 'preview_rejected'
  | 'rate_limited'
  | 'remote_spending_disabled'
  | 'remote_spending_revoked'
  | 'role_required'
  | 'session_revoked'
  | 'stripe_unavailable'
  | 'temporarily_unavailable'
  | 'upgrade_cap_exceeded'
  | 'validation_failed'

/**
 * What the wire actually carries: a known code, or an unknown future one
 * (e.g. the NAS W3 card-health family). The `(string & {})` arm keeps unknown
 * codes assignable — consumers must keep an unknown-code fallback branch.
 */
export type BillingRefusalCode = KnownBillingRefusalCode | (string & {})

/**
 * The closed set of terminal reasons a settled-poll charge can fail with (NAS
 * `cli-charge-failure-reason.ts` — all four values), plus the raw Stripe code
 * NAS pre-#711 leaks for SCA-on-upgrade.
 */
export type KnownChargeFailureReason =
  | 'authentication_required'
  | 'card_declined'
  | 'payment_method_expired'
  | 'processing_error'
  | 'subscription_payment_intent_requires_action'

/** Wire shape: a known reason or an unknown future one; degrade safely. */
export type ChargeFailureReason = KnownChargeFailureReason | (string & {})

/**
 * The fields the client reads out of an error envelope's opaque `payload`
 * (`_serialize_billing_error` forwards the raw NAS body; the contract types it
 * as `JsonValue`). Notably `remainingUsd` on `monthly_cap_exceeded`, so the
 * client can render the same detail the CLI does.
 */
export interface BillingErrorPayload {
  isDefaultCeiling?: boolean
  remainingUsd?: string
}
