import type { BillingApi, BillingResult } from './api'
import type {
  BillingStateResponse,
  SubscriptionPreviewResponse,
  SubscriptionStateResponse
} from './types'

/** The shape of one `billingDevFixtures` entry — a canned billing + subscription pair. */
export interface SimulatedFixture {
  billing: BillingResult<BillingStateResponse>
  subscription: BillingResult<SubscriptionStateResponse>
}

// A visible-but-brief pause so the live fixture loop actually sees the "Checking…" /
// "Scheduling…" / "Undoing…" transitions rather than an instant flip.
const SIMULATED_DELAY_MS = 300

const delay = (ms: number) => new Promise<void>(resolve => setTimeout(resolve, ms))

const ok = <T>(data: T): BillingResult<T> => ({ data, ok: true })

/**
 * A fully in-memory BillingApi for DEV fixtures — no gateway. Fetches serve a mutable
 * copy of the fixture, and the subscription-change mutations WRITE that copy's pending
 * state, so the fixture click-through genuinely progresses: schedule sets a pending
 * downgrade (→ the plan card's "Changes to …" + Undo, and the grid's Scheduled marker
 * on refetch), and resume clears any pending downgrade OR cancellation. Consumers reach
 * it transparently via `useBillingApi` (overridden by BillingApiProvider), so no code
 * outside this file is fixture-aware.
 */
export function createSimulatedBillingApi(fixture: SimulatedFixture): BillingApi {
  const billing = fixture.billing
  // Mutable copy so scheduling/undo don't leak back into the shared fixture object.
  let subscription: BillingResult<SubscriptionStateResponse> = structuredClone(fixture.subscription)

  const patchCurrent = (patch: Partial<NonNullable<SubscriptionStateResponse['current']>>) => {
    if (subscription.ok && subscription.data.current) {
      subscription = ok({ ...subscription.data, current: { ...subscription.data.current, ...patch } })
    }
  }

  const tierName = (tierId: string): null | string =>
    (subscription.ok ? subscription.data.tiers.find(tier => tier.tier_id === tierId)?.name : null) ?? null

  return {
    charge: async (_amountUsd, idempotencyKey = 'sim-key') => ({
      data: { charge_id: 'sim-charge', ok: true },
      idempotencyKey,
      ok: true
    }),
    chargeStatus: async () => ok({ amount_usd: '0', ok: true, settled_at: null, status: 'settled' }),
    fetchBillingState: async () => billing,
    fetchSubscriptionState: async () => subscription,
    previewSubscriptionChange: async tierId => {
      await delay(SIMULATED_DELAY_MS)

      const preview: SubscriptionPreviewResponse = {
        effect: 'scheduled',
        effective_at: subscription.ok ? (subscription.data.current?.cycle_ends_at ?? null) : null,
        ok: true,
        target_tier_name: tierName(tierId)
      }

      return ok(preview)
    },
    resumeSubscription: async () => {
      await delay(SIMULATED_DELAY_MS)
      // Undo either scheduled change kind.
      patchCurrent({
        cancel_at_period_end: false,
        cancellation_effective_at: null,
        cancellation_effective_display: null,
        pending_downgrade_at: null,
        pending_downgrade_display: null,
        pending_downgrade_tier_name: null
      })

      return ok({ message: 'Change cancelled.', ok: true })
    },
    scheduleSubscriptionChange: async tierId => {
      await delay(SIMULATED_DELAY_MS)
      patchCurrent({
        pending_downgrade_at: subscription.ok ? (subscription.data.current?.cycle_ends_at ?? null) : null,
        pending_downgrade_display: null,
        pending_downgrade_tier_name: tierName(tierId)
      })

      return ok({ message: 'Downgrade scheduled.', ok: true })
    },
    stepUp: async () => ok({ granted: true, ok: true }),
    updateAutoReload: async () => ok({ ok: true })
  }
}
