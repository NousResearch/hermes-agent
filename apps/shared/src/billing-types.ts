/**
 * Shared Remote Spending wire contracts.
 *
 * These shapes round-trip between the Python tui_gateway and TypeScript clients
 * such as the TUI and desktop app. Keep rendering state, client logic, and the
 * gateway event union out of this runtime-free module.
 */

// ── Remote Spending (Phase 2b) ───────────────────────────────────────

/** One serialized usage bar (mirrors server `_serialize_usage_bar`). */
export interface UsageBarData {
  kind: 'plan' | 'topup'
  remaining_display: string
  total_display: string
  spent_display: string
  pct_used: null | number
  fill_fraction: number
}

/** The shared dollar usage model (mirrors server `_serialize_usage_model`). */
export interface UsageModelData {
  available: boolean
  status?: string
  plan_name?: null | string
  renews_at?: null | string
  renews_display?: null | string
  subscription_remaining_display?: null | string
  topup_remaining_display?: null | string
  total_spendable_display?: null | string
  has_topup?: boolean
  plan_bar?: null | UsageBarData
  topup_bar?: null | UsageBarData
}

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

export interface BillingCardInfo {
  brand: string
  last4: string
  masked: string
  /** "Visa ····4242 — the card on your subscription" (= masked when provenance unknown). */
  display?: string
  /** Raw card-resolution rung ("subPin" | "customerDefault" | "autoRefill") or null on older NAS. */
  resolved_via?: null | string
}

export interface BillingMonthlyCap {
  is_default_ceiling: boolean
  limit_display: string
  limit_usd: string | null
  spent_display: string
  spent_this_month_usd: string | null
}

export interface BillingAutoReload {
  card:
    | { kind: 'canonical' }
    | {
        kind: 'distinct'
        payment_method_id: string
        brand: string | null
        last4: string | null
      }
    | { kind: 'none' }
  enabled: boolean
  reload_to_display: string
  reload_to_usd: string | null
  threshold_display: string
  threshold_usd: string | null
}

export interface BillingStateResponse {
  auto_reload: BillingAutoReload | null
  balance_display: string
  balance_usd: string | null
  // NAS capability (canChangePlan) when the server sends it; legacy role fallback otherwise
  can_change_plan?: boolean
  can_charge: boolean
  card: BillingCardInfo | null
  charge_presets: string[]
  charge_presets_display: string[]
  cli_billing_enabled: boolean
  error?: string | null
  is_admin: boolean
  logged_in: boolean
  max_usd: string | null
  min_usd: string | null
  monthly_cap: BillingMonthlyCap | null
  ok: boolean
  org_name: string | null
  portal_url: string | null
  role: string | null
  // Shared dollar usage model (two-bar view), embedded by the gateway so /topup
  // renders the same bars as /usage and /subscription from this single fetch.
  usage?: UsageModelData
}

/**
 * Raw error payload echoed from the server (`_serialize_billing_error`). Carries
 * the extra fields a few error codes attach — notably `remainingUsd` on
 * `monthly_cap_exceeded` — so the client can render the same detail the CLI does.
 */
export interface BillingErrorPayload {
  isDefaultCeiling?: boolean
  remainingUsd?: string
}

export interface BillingChargeResponse {
  actor?: string
  charge_id?: string
  code?: string
  error?: BillingRefusalCode
  idempotency_key?: string
  message?: string
  ok: boolean
  payload?: BillingErrorPayload
  portal_url?: string | null
  recovery?: string
  retry_after?: number | null
}

export interface BillingChargeStatusResponse {
  amount_usd?: string | null
  error?: BillingRefusalCode
  message?: string
  ok: boolean
  payload?: BillingErrorPayload
  portal_url?: string | null
  reason?: ChargeFailureReason | null
  retry_after?: number | null
  settled_at?: string | null
  status?: string
}

export interface BillingMutationResponse {
  actor?: string
  code?: string
  error?: BillingRefusalCode
  granted?: boolean
  message?: string
  ok: boolean
  /**
   * On ok:false, the structured error payload. On ok:true the gateway passes
   * through the raw NAS success body (e.g. rail, changeType, cancelAtPeriodEnd
   * for subscription mutations), which has no stable shape here.
   */
  payload?: BillingErrorPayload | Record<string, unknown>
  portal_url?: string | null
  recovery?: string
  retry_after?: number | null
}

export interface SubscriptionTierOption {
  tier_id: string
  name: string
  tier_order: number                  // sorts the picker + upgrade/downgrade hint
  dollars_per_month_display: string   // pre-formatted ($X / $X.YY)
  monthly_credits: string | null
  is_current: boolean                 // the active plan: shown, not selectable
  is_enabled: boolean                 // false = grandfathered current tier
}

export interface SubscriptionStateResponse {
  ok: boolean
  logged_in: boolean
  is_admin: boolean
  can_change_plan: boolean        // NAS capability (canChangePlan: OWNER/ADMIN/FINANCE_ADMIN); legacy role fallback when the server omits it
  org_name: string | null
  org_id: string | null           // org.id from the NAS response
  role: string | null
  context: 'personal' | 'team'   // personal account vs team/org terminal
  current: {
    tier_id: string | null        // null = free (no active sub)
    tier_name: string | null
    monthly_credits: string | null
    credits_remaining: string | null
    cycle_ends_at: string | null  // ISO
    pending_downgrade_tier_name: string | null
    pending_downgrade_at: string | null
    pending_downgrade_display: string | null  // formatted pending_downgrade_at
    cancel_at_period_end: boolean // subscription scheduled to cancel at period end
    cancellation_effective_at: string | null  // ISO when cancellation takes effect
    cancellation_effective_display: string | null  // formatted cancellation_effective_at
  } | null
  tiers: SubscriptionTierOption[]  // selectable catalog for the in-terminal picker
  portal_url: string | null
  error?: string | null
  // Shared dollar usage model (two-bar view), embedded by the gateway so the
  // overlay renders the same bars as /usage from this single fetch.
  usage?: UsageModelData
}

// A chargeless quote (POST /subscription/preview) of what a change would do.
// `effect` drives the confirm copy; a failed preview reuses the typed-error
// envelope fields (same as the mutations) so a 403 still triggers the step-up.
export interface SubscriptionPreviewResponse {
  ok: boolean
  effect?: 'charge_now' | 'scheduled' | 'no_op' | 'blocked'
  reason?: string | null
  current_tier_id?: string | null
  current_tier_name?: string | null
  target_tier_id?: string | null
  target_tier_name?: string | null
  monthly_credits_delta?: string | null
  amount_due_now_cents?: number | null  // the prorated upfront charge for an upgrade
  effective_at?: string | null          // ISO, when a scheduled change lands
  // typed-error envelope (present when ok=false)
  error?: BillingRefusalCode
  message?: string
  portal_url?: string | null
  retry_after?: number | null
  payload?: BillingErrorPayload
  actor?: string
  code?: string
  recovery?: string
}

// The single money route (POST /subscription/upgrade). `status` distinguishes a
// completed upgrade from an SCA/decline that must finish in the portal at
// `recovery_url`. `idempotency_key` is echoed so a retry reuses it.
export interface SubscriptionUpgradeResponse {
  ok: boolean
  status?: 'upgraded' | 'already_on_tier' | 'requires_action' | 'payment_failed'
  target_tier_name?: string | null
  recovery_url?: string | null
  reason?: ChargeFailureReason | null
  idempotency_key?: string
  // typed-error envelope (present when ok=false)
  error?: BillingRefusalCode
  message?: string
  portal_url?: string | null
  retry_after?: number | null
  payload?: BillingErrorPayload
  actor?: string
  code?: string
  recovery?: string
}
