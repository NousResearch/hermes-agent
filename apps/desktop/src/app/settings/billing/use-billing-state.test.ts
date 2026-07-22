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
import { buildManageSubscriptionUrl, deriveBillingView, formatMonthlyCreditsDelta } from './use-billing-state'

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
    expect(view.topupRow?.description).toBe(
      "Remote spending is off for this account — a billing admin can turn it on from the portal's Hermes Agent page."
    )
    expect(view.topupRow?.chips).toBeUndefined()
    expect(view.refillRow).toMatchObject({
      action: { label: 'Manage' },
      description: 'Charges $10 automatically when your balance falls below $5.',
      manageInApp: true,
      pill: { label: 'Enabled', tone: 'primary' }
    })
    expect(view.usageRows.map(row => row.id)).toEqual(['subscription_credits', 'topup_credits', 'monthly_cap'])
  })

  it('derives the post-train shape with card provenance, presets, and denominated usage bars', () => {
    const view = deriveBillingView(okBilling(postTrainBillingState), okSubscription(postTrainSubscriptionState))

    expect(view.status).toBe('normal')
    expect(view.paymentRow?.value).toBe('Visa •••• 4242 - subscription card')
    expect(view.topupRow?.chips?.map(chip => chip.label)).toEqual(['$25', '$50', '$100'])
    expect(view.plan?.link?.url).toBe('https://portal.nousresearch.com/manage-subscription?org_id=org_123')
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

    expect(view.refillRow?.caption).toContain('Mastercard ••4444')
    expect(view.refillRow?.caption).toContain('reconcile')
    expect(view.refillRow?.action).toEqual({
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

    expect(view.refillRow?.caption).toContain('a different card')
    expect(view.refillRow?.caption).not.toContain('null')
    expect(view.refillRow?.action?.url).toBe('https://portal.nousresearch.com/billing')
  })

  it('renders the normal enabled auto-refill row when the card is null (no crash)', () => {
    // The gateway emits auto_reload.card: null for a missing/unknown-kind card.
    const view = deriveBillingView(
      okBilling({ ...todayBillingState, auto_reload: { ...todayBillingState.auto_reload, card: null } }),
      okSubscription(todaySubscriptionState)
    )

    expect(view.refillRow).toMatchObject({
      action: { label: 'Manage' },
      description: 'Charges $10 automatically when your balance falls below $5.',
      manageInApp: true,
      pill: { label: 'Enabled', tone: 'primary' }
    })
  })

  it('keeps buy credit controls visible but disabled when no card is on file', () => {
    const fixture = billingDevFixtures['no-card']
    const view = deriveBillingView(fixture.billing, fixture.subscription)
    const buyCredits = view.topupRow

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
    expect(view.paymentRow).toBeUndefined()
    expect(view.topupRow).toBeUndefined()
    expect(view.refillRow).toBeUndefined()
    expect(view.usageRows).toEqual([])
  })

  it('derives a refusal notice when billing.state is unavailable', () => {
    const view = deriveBillingView(endpointUnavailableBilling, okSubscription(todaySubscriptionState))

    expect(view.status).toBe('refusal')
    expect(view.summary.map(item => item.value)).toEqual(['—', '—', '—'])
    expect(view.notice).toMatchObject({
      title: 'Billing endpoint unavailable'
    })
    expect(view.paymentRow).toBeUndefined()
    expect(view.topupRow).toBeUndefined()
    expect(view.refillRow).toBeUndefined()
  })

  it('keeps subscription unavailable as a plan-card degradation with a live portal link', () => {
    const view = deriveBillingView(okBilling(todayBillingState), endpointUnavailableSubscription)

    expect(view.status).toBe('normal')
    expect(view.plan).toMatchObject({
      caption: 'Subscription details are unavailable; opening the portal is still available.',
      tierName: 'Ultra'
    })
    expect(view.plan?.action).toBeUndefined()
    // The caption promises the portal is still reachable — so the link must exist.
    expect(view.plan?.link).toMatchObject({
      label: 'Adjust plan ↗',
      url: 'https://portal.nousresearch.com/manage-subscription'
    })
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

  it('renders top-up balance as a bare amount — no bar (no denominator exists)', () => {
    const view = deriveBillingView(okBilling(postTrainBillingState), okSubscription(postTrainSubscriptionState))
    const topup = view.usageRows.find(row => row.id === 'topup_credits')

    expect(topup?.value).toBe('$75')
    expect(topup?.bar).toBeUndefined()
  })

  it('renders zero top-up balance without a bar too', () => {
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

    const topup = view.usageRows.find(row => row.id === 'topup_credits')

    expect(topup?.value).toBe('$0')
    expect(topup?.bar).toBeUndefined()
  })
})

describe('derivePlanCard (current-plan card)', () => {
  it('offers an in-app "View plans" button for a free personal account that can change plans', () => {
    const fixture = billingDevFixtures['free-personal']
    const view = deriveBillingView(fixture.billing, fixture.subscription)

    expect(view.plan).toMatchObject({ action: { label: 'View plans' }, tierName: 'Free' })
    expect(view.plan?.link).toBeUndefined()
  })

  it('offers an in-app "Change plan" button for a personal subscriber', () => {
    const fixture = billingDevFixtures['subscriber-personal']
    const view = deriveBillingView(fixture.billing, fixture.subscription)

    expect(view.plan).toMatchObject({ action: { label: 'Change plan' }, price: '$20', tierName: 'Plus' })
    expect(view.plan?.link).toBeUndefined()
  })

  it('gives teams a portal escape hatch and no in-app button', () => {
    // todaySubscriptionState is context: 'team'.
    const view = deriveBillingView(okBilling(todayBillingState), okSubscription(todaySubscriptionState))

    expect(view.plan?.action).toBeUndefined()
    expect(view.plan?.link).toMatchObject({
      label: 'Adjust plan ↗',
      url: 'https://portal.nousresearch.com/manage-subscription?org_id=sid-5'
    })
  })

  it('gives non-changing members a portal link but no in-app button', () => {
    const view = deriveBillingView(
      okBilling(todayBillingState),
      okSubscription({ ...todaySubscriptionState, can_change_plan: false, context: 'personal' })
    )

    expect(view.plan?.action).toBeUndefined()
    expect(view.plan?.link).toMatchObject({
      label: 'Adjust plan ↗',
      url: 'https://portal.nousresearch.com/manage-subscription?org_id=sid-5'
    })
  })

  it('withholds the in-app button (no dead click) and offers the portal link when nothing is actionable', () => {
    // A subscriber whose only enabled tier is the one they are already on: the grid
    // would show a single inert card, so the "Change plan" button must not appear.
    const view = deriveBillingView(
      okBilling(todayBillingState),
      okSubscription({
        ...todaySubscriptionState,
        can_change_plan: true,
        context: 'personal',
        current: { ...todaySubscriptionState.current, tier_id: 'solo', tier_name: 'Solo' },
        tiers: [
          {
            dollars_per_month_display: '$10',
            is_current: true,
            is_enabled: true,
            monthly_credits: '10',
            name: 'Solo',
            tier_id: 'solo',
            tier_order: 0
          }
        ]
      })
    )

    expect(view.tiers.map(tier => tier.state)).toEqual(['current'])
    expect(view.plan?.action).toBeUndefined()
    expect(view.plan?.link?.label).toBe('Adjust plan ↗')
  })

  it('gives a top-tier subscriber a portal link, not a dead in-app button', () => {
    // On the highest tier, every enabled tile below is a downgrade — but downgrades are
    // themselves actionable in-app at ticket 11, so this stays a "Change plan" account.
    // (The dead-grid case is a subscriber whose only tile is `current`; covered above.)
    const view = deriveBillingView(
      okBilling(todayBillingState),
      okSubscription({
        ...todaySubscriptionState,
        can_change_plan: true,
        context: 'personal',
        current: { ...todaySubscriptionState.current, tier_id: 'top', tier_name: 'Ultra' },
        tiers: [
          {
            dollars_per_month_display: '$0',
            is_current: false,
            is_enabled: true,
            monthly_credits: '0.1',
            name: 'Free',
            tier_id: 't_free',
            tier_order: 0
          },
          {
            dollars_per_month_display: '$200',
            is_current: true,
            is_enabled: true,
            monthly_credits: '220',
            name: 'Ultra',
            tier_id: 'top',
            tier_order: 1
          }
        ]
      })
    )

    expect(view.tiers.some(tier => tier.state === 'upgrade')).toBe(false)
    // Free below is an in-app downgrade → still actionable → in-app button.
    expect(view.plan?.action).toMatchObject({ label: 'Change plan' })
  })

  it('surfaces a scheduled downgrade as the plan-card pending state (drives the undo)', () => {
    const fixture = billingDevFixtures['pending-downgrade']
    const view = deriveBillingView(fixture.billing, fixture.subscription)

    expect(view.plan).toMatchObject({
      action: { label: 'Change plan' },
      caption: 'Changes to Free on Aug 15.',
      pending: { kind: 'downgrade', tierName: 'Free', when: 'Aug 15' },
      tierName: 'Plus'
    })
  })

  it('surfaces a scheduled cancellation as "Cancels on …" with the same undo and no grid marker', () => {
    const fixture = billingDevFixtures['pending-cancellation']
    const view = deriveBillingView(fixture.billing, fixture.subscription)

    expect(view.plan).toMatchObject({
      action: { label: 'Change plan' },
      caption: 'Cancels on Aug 15.',
      pending: { kind: 'cancellation', when: 'Aug 15' },
      tierName: 'Plus'
    })
    // A cancellation has no target tier → nothing to mark in the grid.
    expect(view.tiers.some(tier => tier.state === 'scheduled')).toBe(false)
  })

  it('lets a pending downgrade win over a cancellation when both are set', () => {
    const view = deriveBillingView(
      okBilling(todayBillingState),
      okSubscription({
        ...todaySubscriptionState,
        can_change_plan: true,
        context: 'personal',
        current: {
          ...todaySubscriptionState.current,
          cancel_at_period_end: true,
          cancellation_effective_at: '2026-09-01T00:00:00Z',
          cancellation_effective_display: 'Sep 1',
          pending_downgrade_at: '2026-08-15T00:00:00Z',
          pending_downgrade_display: 'Aug 15',
          pending_downgrade_tier_name: 'Free',
          tier_id: 'plus',
          tier_name: 'Plus'
        },
        tiers: [
          {
            dollars_per_month_display: '$0',
            is_current: false,
            is_enabled: true,
            monthly_credits: '0.1',
            name: 'Free',
            tier_id: 'free',
            tier_order: 0
          },
          {
            dollars_per_month_display: '$20',
            is_current: true,
            is_enabled: true,
            monthly_credits: '22',
            name: 'Plus',
            tier_id: 'plus',
            tier_order: 1
          }
        ]
      })
    )

    // Downgrade wins (names a concrete target); card + grid agree on it.
    expect(view.plan?.pending).toMatchObject({ kind: 'downgrade', tierName: 'Free' })
    expect(view.plan?.caption).toBe('Changes to Free on Aug 15.')
    expect(view.tiers.find(tier => tier.name === 'Free')?.state).toBe('scheduled')
  })

  it('offers only the portal link when the tier catalog is empty', () => {
    const view = deriveBillingView(
      okBilling(todayBillingState),
      okSubscription({
        ...todaySubscriptionState,
        can_change_plan: true,
        context: 'personal',
        current: null,
        tiers: []
      })
    )

    expect(view.tiers).toEqual([])
    expect(view.plan?.action).toBeUndefined()
    expect(view.plan?.link?.label).toBe('Adjust plan ↗')
  })
})

describe('derivePlanTiers (plans grid)', () => {
  it('marks the current tier, upgrades, and in-app downgrades for a subscriber', () => {
    const fixture = billingDevFixtures['subscriber-personal']
    const view = deriveBillingView(fixture.billing, fixture.subscription)
    const byName = Object.fromEntries(view.tiers.map(tier => [tier.name, tier]))

    expect(view.tiers.map(tier => tier.name)).toEqual(['Free', 'Plus', 'Super', 'Ultra'])
    expect(byName.Free.state).toBe('downgrade')
    // Downgrades act in-app (no portal URL / caption) — the PlanCard wires the confirm flow.
    expect('action' in byName.Free).toBe(false)
    expect(byName.Plus.state).toBe('current')
    expect('action' in byName.Plus).toBe(false)
    expect(byName.Super).toMatchObject({
      action: {
        url: 'https://portal.nousresearch.com/manage-subscription?org_id=org_personal_plus&plan=cltier222super222personal'
      },
      creditsDisplay: '$110 credits/mo',
      state: 'upgrade'
    })
    expect(byName.Ultra.state).toBe('upgrade')
  })

  it('marks the pending downgrade target "scheduled" (inert) while other tiers stay actionable', () => {
    const fixture = billingDevFixtures['pending-downgrade']
    const view = deriveBillingView(fixture.billing, fixture.subscription)
    const byName = Object.fromEntries(view.tiers.map(tier => [tier.name, tier]))

    // Free is the scheduled target → inert marker, not another "Downgrade".
    expect(byName.Free.state).toBe('scheduled')
    expect('action' in byName.Free).toBe(false)
    expect(byName.Plus.state).toBe('current')
    // Reschedule stays possible on the other lower/higher tiers.
    expect(byName.Super.state).toBe('upgrade')
    expect(byName.Ultra.state).toBe('upgrade')
  })

  it('marks the free/lowest tier current (inert) and every paid tier an upgrade when there is no subscription', () => {
    const fixture = billingDevFixtures['free-personal']
    const view = deriveBillingView(fixture.billing, fixture.subscription)
    const byName = Object.fromEntries(view.tiers.map(tier => [tier.name, tier]))

    // No "subscribe to Free" — the $0 tier is the current plan, not a choice.
    expect(view.tiers.map(tier => tier.state)).toEqual(['current', 'upgrade', 'upgrade', 'upgrade'])
    expect('action' in byName.Free).toBe(false)
    // No downgrade state can exist without a subscription.
    expect(view.tiers.some(tier => tier.state === 'downgrade')).toBe(false)
    expect(byName.Plus).toMatchObject({
      action: {
        url: 'https://portal.nousresearch.com/manage-subscription?org_id=org_personal_free&plan=cltier111plus1111personal'
      }
    })
  })

  it('still lists a tier whose name has no art mapping (text-only card, no layout break)', () => {
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
            monthly_credits: '0.1',
            name: 'Free',
            tier_id: 'cltier_free_0000',
            tier_order: 0
          },
          {
            dollars_per_month_display: '$5',
            is_current: false,
            is_enabled: true,
            monthly_credits: '5',
            name: 'Mystery',
            tier_id: 'cltier_mystery_0000',
            tier_order: 5
          }
        ]
      })
    )

    // The unknown-named paid tier still lists (art resolves to null → text-only).
    expect(view.tiers.map(tier => tier.name)).toEqual(['Free', 'Mystery'])
    expect(view.tiers.find(tier => tier.name === 'Mystery')?.state).toBe('upgrade')
  })

  it('keeps a grandfathered (is_enabled:false) CURRENT tier inert and orders downgrades against it', () => {
    // NAS marks a grandfathered current tier is_enabled:false. It must still appear
    // (inert "Current plan") and define the order boundary — lower enabled tiers are
    // downgrades, higher ones are Choose.
    const view = deriveBillingView(
      okBilling(todayBillingState),
      okSubscription({
        ...todaySubscriptionState,
        context: 'personal',
        current: { ...todaySubscriptionState.current, tier_id: 'legacy_mid', tier_name: 'Legacy' },
        tiers: [
          {
            dollars_per_month_display: '$5',
            is_current: false,
            is_enabled: true,
            monthly_credits: '5',
            name: 'Basic',
            tier_id: 'basic',
            tier_order: 0
          },
          {
            dollars_per_month_display: '$15',
            is_current: true,
            is_enabled: false,
            monthly_credits: '15',
            name: 'Legacy',
            tier_id: 'legacy_mid',
            tier_order: 1
          },
          {
            dollars_per_month_display: '$40',
            is_current: false,
            is_enabled: true,
            monthly_credits: '40',
            name: 'Ultra',
            tier_id: 'ultra',
            tier_order: 2
          }
        ]
      })
    )

    const byName = Object.fromEntries(view.tiers.map(tier => [tier.name, tier]))

    expect(view.tiers.map(tier => tier.name)).toEqual(['Basic', 'Legacy', 'Ultra'])
    expect(byName.Legacy.state).toBe('current')
    expect('action' in byName.Legacy).toBe(false)
    expect(byName.Basic.state).toBe('downgrade')
    expect('action' in byName.Basic).toBe(false)
    expect(byName.Ultra).toMatchObject({ action: { label: 'Choose ↗' }, state: 'upgrade' })
  })

  it('backs Choose URLs with billing.portal_url (org_id + plan intact) when the subscription has no portal_url', () => {
    const view = deriveBillingView(
      okBilling({ ...todayBillingState, portal_url: 'https://billing.example.com/x' }),
      okSubscription({
        ...todaySubscriptionState,
        can_change_plan: true,
        context: 'personal',
        current: null,
        portal_url: null,
        tiers: [
          {
            dollars_per_month_display: '$0',
            is_current: false,
            is_enabled: true,
            monthly_credits: '0.1',
            name: 'Free',
            tier_id: 'free0',
            tier_order: 0
          },
          {
            dollars_per_month_display: '$20',
            is_current: false,
            is_enabled: true,
            monthly_credits: '22',
            name: 'Plus',
            tier_id: 'plus1',
            tier_order: 1
          }
        ]
      })
    )

    expect(view.tiers.find(tier => tier.name === 'Plus')).toMatchObject({
      action: { url: 'https://billing.example.com/manage-subscription?org_id=sid-5&plan=plus1' }
    })
  })

  it('drops grandfathered (is_enabled: false) tiers from the grid', () => {
    const view = deriveBillingView(
      okBilling(todayBillingState),
      okSubscription({
        ...todaySubscriptionState,
        context: 'personal',
        current: null,
        tiers: [
          {
            dollars_per_month_display: '$20',
            is_current: false,
            is_enabled: true,
            monthly_credits: '22',
            name: 'Plus',
            tier_id: 'plus',
            tier_order: 1
          },
          {
            dollars_per_month_display: '$9',
            is_current: false,
            is_enabled: false,
            monthly_credits: '9',
            name: 'Legacy',
            tier_id: 'legacy',
            tier_order: 1
          }
        ]
      })
    )

    expect(view.tiers.map(tier => tier.name)).toEqual(['Plus'])
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

  it('appends plan=<tierId> when a tier is chosen', () => {
    expect(
      buildManageSubscriptionUrl(
        { org_id: 'org_123', portal_url: 'https://portal.nousresearch.com/billing' },
        null,
        'tier_abc'
      )
    ).toBe('https://portal.nousresearch.com/manage-subscription?org_id=org_123&plan=tier_abc')
  })

  it('omits the plan param when no tier is given', () => {
    expect(
      buildManageSubscriptionUrl({ org_id: 'org_123', portal_url: 'https://portal.nousresearch.com/billing' }, null)
    ).toBe('https://portal.nousresearch.com/manage-subscription?org_id=org_123')
  })

  it('applies org_id + plan to the hard-coded portal fallback when no portal_url resolves', () => {
    // Regression: the fallback must be the last-resort ORIGIN, not a bare return that
    // silently drops org_id/plan.
    expect(buildManageSubscriptionUrl({ org_id: 'org_z', portal_url: null }, null, 'tier_q')).toBe(
      'https://portal.nousresearch.com/manage-subscription?org_id=org_z&plan=tier_q'
    )
  })
})

describe('formatMonthlyCreditsDelta', () => {
  it('renders a bare negative decimal as signed dollars, never a raw number', () => {
    // Credits are DOLLARS — "-88" must not render bare.
    expect(formatMonthlyCreditsDelta('-88')).toBe('−$88/mo')
  })

  it('renders a positive delta with a plus sign and dollar formatting', () => {
    expect(formatMonthlyCreditsDelta('40')).toBe('+$40/mo')
    expect(formatMonthlyCreditsDelta('-12.50')).toBe('−$12.50/mo')
  })

  it('hides the line (null) for a zero or absent delta', () => {
    expect(formatMonthlyCreditsDelta('0')).toBeNull()
    expect(formatMonthlyCreditsDelta(null)).toBeNull()
    expect(formatMonthlyCreditsDelta(undefined)).toBeNull()
    expect(formatMonthlyCreditsDelta('')).toBeNull()
  })
})
