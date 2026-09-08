import { describe, expect, it } from 'vitest'

import type { BillingStateResponse, SubscriptionStateResponse } from './types'

const fullBillingState = {
  auto_reload: {
    card: { kind: 'canonical' },
    enabled: true,
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
  is_admin: true,
  logged_in: true,
  max_usd: '10000',
  min_usd: '10',
  monthly_cap: {
    is_default_ceiling: false,
    limit_display: '$1,000',
    limit_usd: '1000',
    spent_display: '$180',
    spent_this_month_usd: '180'
  },
  ok: true,
  org_name: 'Acme Research',
  portal_url: 'https://portal.nousresearch.com/billing',
  role: 'OWNER',
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

const deployedTodayBillingState = {
  auto_reload: null,
  balance_display: '$0.00',
  balance_usd: null,
  can_charge: false,
  card: {
    brand: 'mastercard',
    last4: '4444',
    masked: 'mastercard ....4444'
  },
  charge_presets: [],
  charge_presets_display: [],
  cli_billing_enabled: false,
  is_admin: true,
  logged_in: true,
  max_usd: null,
  min_usd: null,
  monthly_cap: null,
  ok: true,
  org_name: 'Fresh Deploy',
  portal_url: null,
  role: 'OWNER'
} satisfies BillingStateResponse

const loggedOutSubscriptionState = {
  can_change_plan: false,
  context: 'personal',
  current: null,
  is_admin: false,
  logged_in: false,
  ok: true,
  org_id: null,
  org_name: null,
  portal_url: 'https://portal.nousresearch.com/login',
  role: null,
  tiers: []
} satisfies SubscriptionStateResponse

describe('desktop billing wire types', () => {
  it('pins realistic billing and subscription RPC payload shapes', () => {
    expect(fullBillingState.card?.resolved_via).toBe('subPin')
    expect(deployedTodayBillingState.can_charge).toBe(false)
    expect(deployedTodayBillingState.cli_billing_enabled).toBe(false)
    expect(deployedTodayBillingState.card?.last4).toBe('4444')
    expect(loggedOutSubscriptionState.logged_in).toBe(false)
  })
})
