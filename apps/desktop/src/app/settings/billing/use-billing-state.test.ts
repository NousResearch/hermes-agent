import { describe, expect, it } from 'vitest'

import {
  billingDevFixtures,
  endpointUnavailableBilling,
  endpointUnavailableSubscription,
  loggedOutBillingState,
  loggedOutSubscriptionState,
  okBilling,
  okSubscription,
  postTrainBillingState,
  postTrainSubscriptionState,
  todayBillingState,
  todaySubscriptionState
} from './fixtures.test-util'
import { buildManageSubscriptionUrl, deriveBillingView } from './use-billing-state'

function usageRowFor(
  fixtureName: keyof typeof billingDevFixtures,
  rowId: 'monthly_cap' | 'subscription_credits' | 'topup_credits'
) {
  const fixture = billingDevFixtures[fixtureName]
  const view = deriveBillingView(fixture.billing, fixture.subscription)

  return view.usageRows.find(row => row.id === rowId)
}

function subscriptionCreditsRowForRemaining(remaining: string) {
  const view = deriveBillingView(
    okBilling(todayBillingState),
    okSubscription({
      ...todaySubscriptionState,
      current: { ...todaySubscriptionState.current, credits_remaining: remaining, monthly_credits: '220' }
    })
  )

  return view.usageRows.find(row => row.id === 'subscription_credits')
}

function monthlyCapRowForSpent(spent: string) {
  const view = deriveBillingView(
    okBilling({
      ...todayBillingState,
      monthly_cap: {
        is_default_ceiling: false,
        limit_display: '$100',
        limit_usd: '100',
        spent_display: `$${spent}`,
        spent_this_month_usd: spent
      }
    }),
    okSubscription(todaySubscriptionState)
  )

  return view.usageRows.find(row => row.id === 'monthly_cap')
}

describe('deriveBillingView', () => {
  it('derives the deployed-today shape with fail-open disabled charge controls', () => {
    const view = deriveBillingView(okBilling(todayBillingState), okSubscription(todaySubscriptionState))

    expect(view.status).toBe('normal')
    expect(view.summary).toContainEqual({ label: 'Balance', value: '$996.47' })
    expect(view.summary).toContainEqual({ label: 'Plan', value: 'Ultra · $200/mo' })
    const buyCredits = view.accountRows.find(row => row.id === 'buy_credits')

    expect(buyCredits?.description).toBe(
      "Remote spending is off for this account — a billing admin can turn it on from the portal's Hermes Agent page."
    )
    expect(buyCredits?.chips).toBeUndefined()
    expect(view.accountRows.find(row => row.id === 'auto_reload')).toMatchObject({
      action: { label: 'Manage' },
      caption: 'Refill $10 when balance falls below $5',
      pill: { label: 'Enabled', tone: 'primary' }
    })
    expect(view.usageRows.map(row => row.id)).toEqual(['subscription_credits', 'topup_credits', 'monthly_cap'])
  })

  it('derives the post-train shape with card provenance, presets, and denominated usage bars', () => {
    const view = deriveBillingView(okBilling(postTrainBillingState), okSubscription(postTrainSubscriptionState))

    expect(view.status).toBe('normal')
    expect(view.accountRows.find(row => row.id === 'payment_method')?.value).toBe('Visa •••• 4242 - subscription card')
    expect(view.accountRows.find(row => row.id === 'buy_credits')?.chips?.map(chip => chip.label)).toEqual([
      '$25',
      '$50',
      '$100'
    ])
    expect(view.accountRows.find(row => row.id === 'subscription')?.action?.url).toBe(
      'https://portal.nousresearch.com/manage-subscription?org_id=org_123'
    )
    expect(view.usageRows.find(row => row.id === 'subscription_credits')).toMatchObject({
      bar: { value: 0.4 },
      value: '$40 of $100 left'
    })
  })

  it('points divergent auto-refill cards at the portal for reconciliation', () => {
    const view = deriveBillingView(
      okBilling({
        ...todayBillingState,
        auto_reload: {
          ...todayBillingState.auto_reload,
          card: { kind: 'distinct', payment_method_id: 'pm_1', brand: 'mastercard', last4: '4444' }
        }
      }),
      okSubscription(todaySubscriptionState)
    )

    const autoReload = view.accountRows.find(row => row.id === 'auto_reload')

    expect(autoReload?.caption).toContain('Mastercard ••4444')
    expect(autoReload?.caption).toContain('reconcile')
    expect(autoReload?.action).toEqual({
      label: 'Reconcile ↗',
      url: 'https://portal.nousresearch.com/billing'
    })
  })

  it('degrades safely when a divergent auto-refill card has no display details', () => {
    const view = deriveBillingView(
      okBilling({
        ...todayBillingState,
        auto_reload: {
          ...todayBillingState.auto_reload,
          card: { kind: 'distinct', payment_method_id: 'pm_1', brand: null, last4: null }
        }
      }),
      okSubscription(todaySubscriptionState)
    )

    const autoReload = view.accountRows.find(row => row.id === 'auto_reload')

    expect(autoReload?.caption).toContain('a different card')
    expect(autoReload?.caption).not.toContain('null')
    expect(autoReload?.action?.url).toBe('https://portal.nousresearch.com/billing')
  })

  it('keeps buy credit controls visible but disabled when no card is on file', () => {
    const fixture = billingDevFixtures['no-card']
    const view = deriveBillingView(fixture.billing, fixture.subscription)
    const buyCredits = view.accountRows.find(row => row.id === 'buy_credits')

    expect(buyCredits).toMatchObject({
      action: { disabled: true, label: 'Buy' },
      description:
        '💳 No saved card for terminal charges yet. Set one up on the portal ' +
        "(one-time credit buys don't save a reusable card)."
    })
    expect(buyCredits?.chips?.map(chip => chip.disabled)).toEqual([true, true, true])
  })

  it('derives a calm logged-out card with no account or usage rows', () => {
    const view = deriveBillingView(okBilling(loggedOutBillingState), okSubscription(loggedOutSubscriptionState))

    expect(view.status).toBe('logged_out')
    expect(view.summary.map(item => item.value)).toEqual(['—', '—', '—'])
    expect(view.notice).toMatchObject({
      title: 'Connect your Nous account'
    })
    expect(view.accountRows).toEqual([])
    expect(view.usageRows).toEqual([])
  })

  it('derives a refusal notice when billing.state is unavailable', () => {
    const view = deriveBillingView(endpointUnavailableBilling, okSubscription(todaySubscriptionState))

    expect(view.status).toBe('refusal')
    expect(view.summary.map(item => item.value)).toEqual(['—', '—', '—'])
    expect(view.notice).toMatchObject({
      title: 'Billing endpoint unavailable'
    })
    expect(view.accountRows).toEqual([])
  })

  it('keeps subscription unavailable as a row-level degradation when billing.state succeeds', () => {
    const view = deriveBillingView(okBilling(todayBillingState), endpointUnavailableSubscription)
    const subscription = view.accountRows.find(row => row.id === 'subscription')

    expect(view.status).toBe('normal')
    expect(subscription).toMatchObject({
      caption: 'Subscription details are unavailable; opening the portal is still available.',
      value: 'Ultra'
    })
  })

  it('free with catalog: tier chips render inline and open the portal', () => {
    const view = deriveBillingView(
      okBilling(todayBillingState),
      okSubscription({
        ...todaySubscriptionState,
        context: 'personal',
        current: null,
        tiers: [
          {
            dollars_per_month_display: '$0',
            is_current: false,
            is_enabled: true,
            monthly_credits: '0',
            name: 'Free',
            tier_id: 'free',
            tier_order: 0
          },
          {
            dollars_per_month_display: '$40',
            is_current: false,
            is_enabled: true,
            monthly_credits: '3000',
            name: 'Ultra',
            tier_id: 'ultra',
            tier_order: 2
          },
          {
            dollars_per_month_display: '$20',
            is_current: false,
            is_enabled: true,
            monthly_credits: '1000',
            name: 'Plus',
            tier_id: 'plus',
            tier_order: 1
          }
        ]
      })
    )

    const subscription = view.accountRows.find(row => row.id === 'subscription')

    expect(subscription?.description).toBe('Paid models need a subscription — pick a plan to start it on the portal.')
    expect(subscription?.chips).toEqual([
      { disabled: false, label: 'Plus · $20/mo · $1,000 credits/mo', url: `${subscription?.action?.url}&plan=plus` },
      { disabled: false, label: 'Ultra · $40/mo · $3,000 credits/mo', url: `${subscription?.action?.url}&plan=ultra` }
    ])
  })

  it('subscriber who can change plans: current tier marked inert, others open the portal', () => {
    const view = deriveBillingView(
      okBilling(todayBillingState),
      okSubscription({
        ...todaySubscriptionState,
        context: 'personal',
        tiers: [
          {
            dollars_per_month_display: '$20',
            is_current: true,
            is_enabled: true,
            monthly_credits: '1000',
            name: 'Plus',
            tier_id: 'plus',
            tier_order: 1
          },
          {
            dollars_per_month_display: '$40',
            is_current: false,
            is_enabled: true,
            monthly_credits: '3000',
            name: 'Ultra',
            tier_id: 'ultra',
            tier_order: 2
          }
        ]
      })
    )

    const subscription = view.accountRows.find(row => row.id === 'subscription')

    expect(subscription?.chips).toEqual([
      { disabled: true, label: '✓ Plus · $20/mo · $1,000 credits/mo' },
      { disabled: false, label: 'Ultra · $40/mo · $3,000 credits/mo', url: `${subscription?.action?.url}&plan=ultra` }
    ])
  })

  it('members and team contexts get no tier chips', () => {
    const member = deriveBillingView(
      okBilling(todayBillingState),
      okSubscription({ ...todaySubscriptionState, can_change_plan: false, context: 'personal' })
    )

    const team = deriveBillingView(okBilling(todayBillingState), okSubscription(todaySubscriptionState))

    expect(member.accountRows.find(row => row.id === 'subscription')?.chips).toBeUndefined()
    expect(team.accountRows.find(row => row.id === 'subscription')?.chips).toBeUndefined()
  })

  it('clamps overdrawn subscription credits to $0 and names the overage', () => {
    const view = deriveBillingView(
      okBilling(todayBillingState),
      okSubscription({
        ...todaySubscriptionState,
        current: { ...todaySubscriptionState.current, credits_remaining: '-0.79', monthly_credits: '220' }
      })
    )

    const row = view.usageRows.find(r => r.id === 'subscription_credits')
    expect(row?.value).toBe('$0 of $220 left · $0.79 over')
    expect(row?.bar?.value).toBe(0)
  })

  it('marks subscription remaining bars as ok above 10% and danger at or below 10%', () => {
    const elevenPercent = subscriptionCreditsRowForRemaining('24.2')

    expect(elevenPercent?.bar?.state).toBe('ok')
    expect(elevenPercent?.bar?.value).toBeCloseTo(0.11)
    expect(usageRowFor('healthy', 'subscription_credits')?.bar).toMatchObject({
      state: 'ok',
      value: 0.6
    })

    // Owner wording is "green until 10%, then red"; the exact 10% boundary is red.
    expect(usageRowFor('boundary', 'subscription_credits')?.bar).toMatchObject({
      state: 'danger',
      value: 0.1
    })

    expect(usageRowFor('low', 'subscription_credits')?.bar).toMatchObject({
      state: 'danger',
      value: 0.09
    })
  })

  it('marks empty or overdrawn subscription bars as danger with a full danger track', () => {
    const row = usageRowFor('empty-overdrawn', 'subscription_credits')

    expect(row?.value).toBe('$0 of $220 left · $0.79 over')
    expect(row?.bar).toMatchObject({
      state: 'danger',
      track: 'danger',
      value: 0
    })
  })

  it('marks monthly cap bars as neutral below 90% and danger at or above 90%', () => {
    expect(usageRowFor('healthy', 'monthly_cap')?.bar).toMatchObject({
      state: 'ok',
      value: 0.89
    })

    expect(monthlyCapRowForSpent('90')?.bar).toMatchObject({
      state: 'danger',
      value: 0.9
    })

    expect(usageRowFor('cap-near', 'monthly_cap')?.bar).toMatchObject({
      state: 'danger',
      value: 0.92
    })

    expect(usageRowFor('cap-hit', 'monthly_cap')?.bar).toMatchObject({
      state: 'danger',
      track: 'danger',
      value: 1
    })
  })

  it('renders top-up balance as a full ok bar when credits remain', () => {
    const view = deriveBillingView(okBilling(postTrainBillingState), okSubscription(postTrainSubscriptionState))

    expect(view.usageRows.find(row => row.id === 'topup_credits')).toMatchObject({
      bar: {
        state: 'ok',
        tone: 'topup',
        value: 1
      },
      value: '$75'
    })
  })

  it('renders zero top-up balance as an empty neutral bar', () => {
    const view = deriveBillingView(
      okBilling({
        ...todayBillingState,
        balance_display: '$0',
        balance_usd: '0',
        usage: {
          ...todayBillingState.usage,
          topup_remaining_display: '$0'
        }
      }),
      undefined
    )

    expect(view.usageRows.find(row => row.id === 'topup_credits')).toMatchObject({
      bar: {
        state: 'neutral',
        tone: 'topup',
        value: 0
      },
      value: '$0'
    })
  })
})

describe('buildManageSubscriptionUrl', () => {
  it('mirrors the TUI manage-subscription URL construction', () => {
    expect(
      buildManageSubscriptionUrl({
        org_id: 'org_123',
        portal_url: 'https://portal.nousresearch.com/billing'
      })
    ).toBe('https://portal.nousresearch.com/manage-subscription?org_id=org_123')
  })

  it('appends the tier as a plan query param when provided', () => {
    expect(
      buildManageSubscriptionUrl(
        {
          org_id: 'org_123',
          portal_url: 'https://portal.nousresearch.com/billing'
        },
        undefined,
        'ultra'
      )
    ).toBe('https://portal.nousresearch.com/manage-subscription?org_id=org_123&plan=ultra')
  })

  it('omits the plan param when no tierId is given', () => {
    expect(
      buildManageSubscriptionUrl(
        { org_id: null, portal_url: 'https://portal.nousresearch.com/billing' },
        undefined,
        undefined
      )
    ).toBe('https://portal.nousresearch.com/manage-subscription')
  })
})
