import type { BillingResult } from './api'
import { postTrainBillingState, subscriberPersonalSubscriptionState } from './dev-fixtures'
import type { BillingStateResponse, SubscriptionStateResponse } from './types'

export {
  billingDevFixtures,
  loggedOutBillingState,
  loggedOutSubscriptionState,
  postTrainBillingState,
  postTrainSubscriptionState,
  todayBillingState,
  todaySubscriptionState
} from './dev-fixtures'

export const okBilling = (data: BillingStateResponse): BillingResult<BillingStateResponse> => ({ data, ok: true })

export const okSubscription = (data: SubscriptionStateResponse): BillingResult<SubscriptionStateResponse> => ({
  data,
  ok: true
})

export const endpointUnavailableBilling = {
  ok: false,
  refusal: {
    kind: 'endpoint_unavailable',
    message: 'Billing endpoint returned a non-JSON response.'
  }
} satisfies BillingResult<BillingStateResponse>

export const endpointUnavailableSubscription = {
  ok: false,
  refusal: {
    kind: 'endpoint_unavailable',
    message: 'Subscription endpoint is not available.'
  }
} satisfies BillingResult<SubscriptionStateResponse>

// A usage-window plan: the dollar `usage` model is unavailable and the page draws the
// Today / This week / This month / Large models gauges instead.
// `month` is optional: an older portal sends no month allowance, so no month row.
function windowPlan(
  name: string,
  {
    day,
    week,
    month,
    topTier,
    paused = false
  }: { day: number; week: number; month?: number; topTier: number; paused?: boolean }
) {
  const now = Date.now()

  const billing = {
    ...postTrainBillingState,
    balance_display: '$12.40',
    balance_usd: '12.40',
    monthly_cap: null,
    org_name: `${name} Fixture`,
    plan_usage: {
      gauges: [
        { label: 'day', left_fraction: day, resets_at_ms: now + 7 * 3_600_000, scope: 'all', style: styleFor(day) },
        {
          label: 'week',
          left_fraction: week,
          resets_at_ms: now + 4 * 86_400_000 + 3_600_000,
          scope: 'all',
          style: styleFor(week)
        },
        ...(month == null
          ? []
          : [
              {
                label: 'month',
                left_fraction: month,
                resets_at_ms: now + 18 * 86_400_000,
                scope: 'all',
                style: styleFor(month)
              }
            ]),
        {
          label: 'top_tier',
          left_fraction: topTier,
          resets_at_ms: now + 7 * 3_600_000,
          scope: 'top_tier',
          style: styleFor(topTier)
        }
      ],
      featured_model_ids: ['example/model-a', 'example/model-c'],
      top_tier_model_ids: ['example/model-c'],
      kind: 'usage_window',
      paused_reason: paused ? 'payment_past_due' : null,
      plan: 'Window Plan'
    },
    usage: { available: false }
  } satisfies BillingStateResponse

  const subscription = {
    ...subscriberPersonalSubscriptionState,
    current: {
      ...subscriberPersonalSubscriptionState.current,
      credits_remaining: null,
      cycle_ends_at: '2026-10-11T08:14:55.000Z',
      monthly_credits: '0',
      tier_id: 'window-plan',
      tier_name: 'Window Plan'
    },
    org_name: `${name} Fixture`,
    plan_usage: billing.plan_usage,
    tiers: [],
    usage: { available: false }
  } satisfies SubscriptionStateResponse

  return { billing: okBilling(billing), subscription: okSubscription(subscription) }
}

// Mirrors hermes_cli/model_picker_usage.py::gauge_style (green > 0.5, yellow >= 0.2, red below).
function styleFor(left: number): string {
  return left < 0.2 ? 'red' : left < 0.5 ? 'yellow' : 'green'
}

export const windowFixtures = {
  'window-healthy': windowPlan('Window Healthy', { day: 0.76, week: 0.93, month: 0.97, topTier: 1 }),
  'window-low': windowPlan('Window Low', { day: 0.12, week: 0.41, topTier: 0 }),
  'window-paused': windowPlan('Window Paused', { day: 0.6, week: 0.8, topTier: 0.5, paused: true })
}
