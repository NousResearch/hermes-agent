import type { BillingResult } from './api'
import type { BillingStateResponse, SubscriptionStateResponse, SubscriptionTierOption } from './types'

const current = (
  overrides: Partial<NonNullable<SubscriptionStateResponse['current']>> = {}
): NonNullable<SubscriptionStateResponse['current']> => ({
  cancel_at_period_end: false,
  cancellation_effective_at: null,
  cancellation_effective_display: null,
  credits_remaining: '120',
  cycle_ends_at: '2026-07-11T08:14:55.000Z',
  monthly_credits: '220',
  pending_downgrade_at: null,
  pending_downgrade_display: null,
  pending_downgrade_tier_name: null,
  tier_id: 'ultra',
  tier_name: 'Ultra',
  ...overrides
})

export const todayBillingState = {
  auto_reload: {
    card: { kind: 'canonical' },
    enabled: true,
    reload_to_display: '$10',
    reload_to_usd: '10',
    threshold_display: '$5',
    threshold_usd: '5'
  },
  balance_display: '$996.47',
  balance_usd: '996.47',
  can_charge: false,
  card: {
    brand: 'visa',
    last4: '3206',
    masked: 'visa ....3206'
  },
  charge_presets: ['100', '250', '500'],
  charge_presets_display: ['$100', '$250', '$500'],
  cli_billing_enabled: false,
  is_admin: true,
  logged_in: true,
  max_usd: '1000',
  min_usd: '10',
  monthly_cap: {
    is_default_ceiling: true,
    limit_display: '$100',
    limit_usd: '100',
    spent_display: '$10',
    spent_this_month_usd: '10'
  },
  ok: true,
  org_name: 'sid-5',
  portal_url: 'https://portal.nousresearch.com/billing',
  role: 'OWNER',
  usage: {
    available: true,
    has_topup: true,
    plan_name: 'Ultra',
    renews_at: '2026-07-11T08:14:55.000Z',
    renews_display: 'Jul 11',
    status: 'active',
    subscription_remaining_display: '$120',
    topup_remaining_display: '$876.47',
    total_spendable_display: '$996.47'
  }
} satisfies BillingStateResponse

export const todaySubscriptionState = {
  can_change_plan: true,
  context: 'team',
  current: current(),
  is_admin: true,
  logged_in: true,
  ok: true,
  org_id: 'sid-5',
  org_name: 'sid-5',
  portal_url: 'https://portal.nousresearch.com/billing',
  role: 'OWNER',
  tiers: [
    {
      dollars_per_month_display: '$200',
      is_current: true,
      is_enabled: true,
      monthly_credits: '220',
      name: 'Ultra',
      tier_id: 'ultra',
      tier_order: 3
    }
  ],
  usage: todayBillingState.usage
} satisfies SubscriptionStateResponse

export const postTrainBillingState = {
  ...todayBillingState,
  auto_reload: {
    card: { kind: 'canonical' },
    enabled: false,
    reload_to_display: '$100',
    reload_to_usd: '100',
    threshold_display: '$25',
    threshold_usd: '25'
  },
  balance_display: '$142.50',
  balance_usd: '142.50',
  can_charge: true,
  card: {
    brand: 'visa',
    display: 'Visa ....4242 - the card on your subscription',
    last4: '4242',
    masked: 'visa ....4242',
    resolved_via: 'subPin'
  },
  charge_presets: ['25', '50', '100'],
  charge_presets_display: ['$25', '$50', '$100'],
  cli_billing_enabled: true,
  monthly_cap: {
    is_default_ceiling: false,
    limit_display: '$1,000',
    limit_usd: '1000',
    spent_display: '$180',
    spent_this_month_usd: '180'
  },
  org_name: 'Acme Research',
  usage: {
    available: true,
    has_topup: true,
    plan_bar: {
      fill_fraction: 0.4,
      kind: 'plan',
      pct_used: 60,
      remaining_display: '$40',
      spent_display: '$60',
      total_display: '$100'
    },
    plan_name: 'Pro',
    renews_at: '2026-07-31T00:00:00Z',
    renews_display: 'Jul 31',
    status: 'active',
    subscription_remaining_display: '$40',
    topup_bar: {
      fill_fraction: 0.75,
      kind: 'topup',
      pct_used: 25,
      remaining_display: '$75',
      spent_display: '$25',
      total_display: '$100'
    },
    topup_remaining_display: '$75',
    total_spendable_display: '$115'
  }
} satisfies BillingStateResponse

export const postTrainSubscriptionState = {
  ...todaySubscriptionState,
  current: current({
    credits_remaining: '40',
    cycle_ends_at: '2026-07-31T00:00:00Z',
    monthly_credits: '100',
    tier_id: 'pro',
    tier_name: 'Pro'
  }),
  org_id: 'org_123',
  org_name: 'Acme Research',
  tiers: [
    {
      dollars_per_month_display: '$20',
      is_current: true,
      is_enabled: true,
      monthly_credits: '100',
      name: 'Pro',
      tier_id: 'pro',
      tier_order: 2
    }
  ],
  usage: postTrainBillingState.usage
} satisfies SubscriptionStateResponse

export const loggedOutBillingState = {
  ...todayBillingState,
  auto_reload: null,
  balance_display: '$0.00',
  balance_usd: null,
  can_charge: false,
  card: null,
  charge_presets: [],
  charge_presets_display: [],
  logged_in: false,
  monthly_cap: null,
  org_name: null,
  portal_url: 'https://portal.nousresearch.com/login',
  role: null,
  usage: undefined
} satisfies BillingStateResponse

export const loggedOutSubscriptionState = {
  ...todaySubscriptionState,
  can_change_plan: false,
  current: null,
  is_admin: false,
  logged_in: false,
  org_id: null,
  org_name: null,
  portal_url: 'https://portal.nousresearch.com/login',
  role: null,
  tiers: [],
  usage: undefined
} satisfies SubscriptionStateResponse

// Full four-tier personal catalog. tier_ids are cuid-like (Prisma) on purpose:
// tier art keys off the lowercase NAME, never the id (ids differ per env). Dollar
// credits are 0.1 / 22 / 110 / 220 to exercise the "$X credits/mo" formatting.
const personalTierCatalog: SubscriptionTierOption[] = [
  {
    dollars_per_month_display: '$0',
    is_current: false,
    is_enabled: true,
    monthly_credits: '0.1',
    name: 'Free',
    tier_id: 'cltier000free0000personal',
    tier_order: 0
  },
  {
    dollars_per_month_display: '$20',
    is_current: false,
    is_enabled: true,
    monthly_credits: '22',
    name: 'Plus',
    tier_id: 'cltier111plus1111personal',
    tier_order: 1
  },
  {
    dollars_per_month_display: '$100',
    is_current: false,
    is_enabled: true,
    monthly_credits: '110',
    name: 'Super',
    tier_id: 'cltier222super222personal',
    tier_order: 2
  },
  {
    dollars_per_month_display: '$200',
    is_current: false,
    is_enabled: true,
    monthly_credits: '220',
    name: 'Ultra',
    tier_id: 'cltier333ultra333personal',
    tier_order: 3
  }
]

const catalogWithCurrent = (currentTierId: null | string): SubscriptionTierOption[] =>
  personalTierCatalog.map(tier => ({ ...tier, is_current: tier.tier_id === currentTierId }))

// Logged-in personal org, no subscription: exercises the "View plans" plan card
// and the full plans grid where every tier is an upgrade.
export const freePersonalBillingState = {
  ...postTrainBillingState,
  balance_display: '$12.00',
  balance_usd: '12.00',
  org_name: 'Personal',
  usage: {
    available: true,
    has_topup: true,
    plan_name: 'Free',
    renews_at: null,
    renews_display: null,
    status: 'active',
    subscription_remaining_display: '$0',
    topup_remaining_display: '$12.00',
    total_spendable_display: '$12.00'
  }
} satisfies BillingStateResponse

export const freePersonalSubscriptionState = {
  ...todaySubscriptionState,
  can_change_plan: true,
  context: 'personal',
  current: null,
  org_id: 'org_personal_free',
  org_name: 'Personal',
  tiers: catalogWithCurrent(null),
  usage: freePersonalBillingState.usage
} satisfies SubscriptionStateResponse

// Personal subscriber on Plus: exercises the "Change plan" plan card, the current
// marker, upgrades (Super/Ultra), and the disabled downgrade (Free).
export const subscriberPersonalBillingState = {
  ...postTrainBillingState,
  org_name: 'Personal',
  usage: {
    ...postTrainBillingState.usage,
    plan_name: 'Plus'
  }
} satisfies BillingStateResponse

export const subscriberPersonalSubscriptionState = {
  ...todaySubscriptionState,
  can_change_plan: true,
  context: 'personal',
  current: current({
    credits_remaining: '12',
    cycle_ends_at: '2026-08-15T00:00:00Z',
    monthly_credits: '22',
    tier_id: 'cltier111plus1111personal',
    tier_name: 'Plus'
  }),
  org_id: 'org_personal_plus',
  org_name: 'Personal',
  tiers: catalogWithCurrent('cltier111plus1111personal'),
  usage: subscriberPersonalBillingState.usage
} satisfies SubscriptionStateResponse

// Personal subscriber on Plus with a downgrade to Free already scheduled at period
// end: exercises the plan-card pending state + undo, and the grid's "Scheduled"
// marker on Free while Super/Ultra stay choosable.
export const pendingDowngradeSubscriptionState = {
  ...subscriberPersonalSubscriptionState,
  current: current({
    credits_remaining: '12',
    cycle_ends_at: '2026-08-15T00:00:00Z',
    monthly_credits: '22',
    pending_downgrade_at: '2026-08-15T00:00:00Z',
    pending_downgrade_display: 'Aug 15',
    pending_downgrade_tier_name: 'Free',
    tier_id: 'cltier111plus1111personal',
    tier_name: 'Plus'
  })
} satisfies SubscriptionStateResponse

// Personal subscriber on Plus with a cancellation (not a downgrade) scheduled at
// period end: exercises the plan-card "Cancels on …" copy + undo, with NO Scheduled
// grid marker (a cancellation has no target tier).
export const pendingCancellationSubscriptionState = {
  ...subscriberPersonalSubscriptionState,
  current: current({
    cancel_at_period_end: true,
    cancellation_effective_at: '2026-08-15T00:00:00Z',
    cancellation_effective_display: 'Aug 15',
    credits_remaining: '12',
    cycle_ends_at: '2026-08-15T00:00:00Z',
    monthly_credits: '22',
    tier_id: 'cltier111plus1111personal',
    tier_name: 'Plus'
  })
} satisfies SubscriptionStateResponse

const okBilling = (data: BillingStateResponse): BillingResult<BillingStateResponse> => ({ data, ok: true })

const okSubscription = (data: SubscriptionStateResponse): BillingResult<SubscriptionStateResponse> => ({
  data,
  ok: true
})

function withUsage(
  name: string,
  {
    autoReload = postTrainBillingState.auto_reload,
    canCharge = true,
    card = postTrainBillingState.card,
    cliBillingEnabled = true,
    monthlyCapSpent = '89',
    remaining,
    subscriptionCurrent = current({ credits_remaining: remaining, monthly_credits: '220' })
  }: {
    autoReload?: BillingStateResponse['auto_reload']
    canCharge?: boolean
    card?: BillingStateResponse['card']
    cliBillingEnabled?: boolean
    monthlyCapSpent?: string
    remaining: string
    subscriptionCurrent?: SubscriptionStateResponse['current']
  }
) {
  const billing = {
    ...postTrainBillingState,
    auto_reload: autoReload,
    balance_display: '$142.50',
    balance_usd: '142.50',
    can_charge: canCharge,
    card,
    cli_billing_enabled: cliBillingEnabled,
    monthly_cap: {
      is_default_ceiling: false,
      limit_display: '$100',
      limit_usd: '100',
      spent_display: `$${monthlyCapSpent}`,
      spent_this_month_usd: monthlyCapSpent
    },
    org_name: `${name} Fixture`,
    usage: {
      ...postTrainBillingState.usage,
      plan_name: 'Ultra',
      subscription_remaining_display: `$${remaining}`,
      total_spendable_display: '$142.50'
    }
  } satisfies BillingStateResponse

  const subscription = {
    ...todaySubscriptionState,
    current: subscriptionCurrent,
    org_name: `${name} Fixture`,
    usage: billing.usage
  } satisfies SubscriptionStateResponse

  return { billing: okBilling(billing), subscription: okSubscription(subscription) }
}

export const billingDevFixtures = {
  healthy: withUsage('Healthy', { monthlyCapSpent: '89', remaining: '132' }),
  'free-personal': {
    billing: okBilling(freePersonalBillingState),
    subscription: okSubscription(freePersonalSubscriptionState)
  },
  'subscriber-personal': {
    billing: okBilling(subscriberPersonalBillingState),
    subscription: okSubscription(subscriberPersonalSubscriptionState)
  },
  'pending-cancellation': {
    billing: okBilling(subscriberPersonalBillingState),
    subscription: okSubscription(pendingCancellationSubscriptionState)
  },
  'pending-downgrade': {
    billing: okBilling(subscriberPersonalBillingState),
    subscription: okSubscription(pendingDowngradeSubscriptionState)
  },
  'auto-refill-divergent': withUsage('Auto Refill Divergent', {
    autoReload: {
      ...postTrainBillingState.auto_reload,
      card: { kind: 'distinct', payment_method_id: 'pm_divergent_1', brand: 'mastercard', last4: '4444' },
      enabled: true
    },
    remaining: '132'
  }),
  low: withUsage('Low', { remaining: '19.8' }),
  boundary: withUsage('Boundary', { remaining: '22' }),
  'empty-overdrawn': withUsage('Empty Overdrawn', { remaining: '-0.79' }),
  'cap-near': withUsage('Cap Near', { monthlyCapSpent: '92', remaining: '132' }),
  'cap-hit': withUsage('Cap Hit', { monthlyCapSpent: '100', remaining: '132' }),
  'no-card': withUsage('No Card', { card: null, remaining: '132' }),
  'no-subscription': withUsage('No Subscription', { remaining: '132', subscriptionCurrent: null }),
  'logged-out': {
    billing: okBilling(loggedOutBillingState),
    subscription: okSubscription(loggedOutSubscriptionState)
  },
  refusal: {
    billing: {
      ok: false,
      refusal: {
        kind: 'temporarily_unavailable',
        message: 'Billing is temporarily unavailable.',
        retryAfter: 90
      }
    },
    subscription: okSubscription(todaySubscriptionState)
  },
  'billing-off': {
    billing: okBilling(todayBillingState),
    subscription: okSubscription(todaySubscriptionState)
  }
} satisfies Record<
  string,
  {
    billing: BillingResult<BillingStateResponse>
    subscription: BillingResult<SubscriptionStateResponse>
  }
>

export type BillingDevFixtureName = keyof typeof billingDevFixtures
