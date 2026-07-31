import { describe, expect, it } from 'vitest'

import { billingDevFixtures } from './dev-fixtures'
import { createSimulatedBillingApi } from './simulated-api'
import { deriveBillingView } from './use-billing-state'

const FREE_TIER_ID = 'cltier000free0000personal'

describe('createSimulatedBillingApi', () => {
  it('progresses the pending state through the whole loop: schedule sets it, resume clears it', async () => {
    const api = createSimulatedBillingApi(billingDevFixtures['subscriber-personal'])
    const billing = await api.fetchBillingState()

    // Baseline: subscriber on Plus, nothing pending.
    const before = deriveBillingView(billing, await api.fetchSubscriptionState())
    expect(before.plan?.pending).toBeUndefined()

    // Schedule a downgrade to Free → the very next fetch shows the pending card + marker.
    expect((await api.scheduleSubscriptionChange(FREE_TIER_ID)).ok).toBe(true)
    const afterSchedule = deriveBillingView(billing, await api.fetchSubscriptionState())
    expect(afterSchedule.plan?.pending).toMatchObject({ kind: 'downgrade', tierName: 'Free' })
    expect(afterSchedule.tiers.find(tier => tier.name === 'Free')?.state).toBe('scheduled')

    // Undo → pending cleared on the next fetch.
    expect((await api.resumeSubscription()).ok).toBe(true)
    const afterResume = deriveBillingView(billing, await api.fetchSubscriptionState())
    expect(afterResume.plan?.pending).toBeUndefined()
    expect(afterResume.tiers.some(tier => tier.state === 'scheduled')).toBe(false)
  })

  it('previews a chargeless scheduled change for the chosen tier', async () => {
    const api = createSimulatedBillingApi(billingDevFixtures['subscriber-personal'])
    const preview = await api.previewSubscriptionChange(FREE_TIER_ID)

    expect(preview).toMatchObject({ data: { effect: 'scheduled', target_tier_name: 'Free' }, ok: true })
  })

  it('undoes a scheduled cancellation too', async () => {
    const api = createSimulatedBillingApi(billingDevFixtures['pending-cancellation'])
    const billing = await api.fetchBillingState()

    expect(deriveBillingView(billing, await api.fetchSubscriptionState()).plan?.pending).toMatchObject({
      kind: 'cancellation'
    })

    await api.resumeSubscription()
    expect(deriveBillingView(billing, await api.fetchSubscriptionState()).plan?.pending).toBeUndefined()
  })

  it('does not mutate the shared fixture object', async () => {
    const api = createSimulatedBillingApi(billingDevFixtures['subscriber-personal'])
    await api.scheduleSubscriptionChange(FREE_TIER_ID)

    const fixture = billingDevFixtures['subscriber-personal']
    expect(deriveBillingView(fixture.billing, fixture.subscription).plan?.pending).toBeUndefined()
  })
})
