/**
 * Where the redesign asks the shared-metrics question (Figma 1335:32952, 1365:34379). The question and
 * its answer are `shared-metrics.ts`'s: the backend's `decided` is the only latch, and the consent host
 * (`components/shared-metrics/consent-dialog.tsx`) reads it only once first-run onboarding is out of
 * the way, so every rule about WHEN to ask holds here unchanged. This module only says WHERE:
 *
 * - `card`: a card at the foot of the chat sidebar, shown wherever the sidebar is.
 * - `tray`: its × put the card away. Not an answer: the question waits in the notifications tray
 *   until it is answered or cleared there, and the mark survives reloads (the guidance ledger).
 * - null: nothing to ask (classic, answered, unknown, an older backend) or cleared from the tray;
 *   Settings still holds the switch.
 *
 * Classic keeps its composer strip and never reads this.
 */

import { computed } from 'nanostores'

import { $guidanceLedger, markGuidance } from './guidance-ledger'
import { $sharedMetricsDetailsOpen, $sharedMetricsOfferConsent, sharedMetricsOfferPending } from './shared-metrics'
import { $portalShell } from './shell-variant'

export type UsageStatsOfferPlace = 'card' | 'tray'

const PLACE_BY_MARK = { deferred: 'tray', cleared: null } as const

export const $usageStatsOfferPlace = computed(
  [$portalShell, $sharedMetricsOfferConsent, $guidanceLedger],
  (portal, consent, ledger): null | UsageStatsOfferPlace => {
    if (!portal || !sharedMetricsOfferPending(consent)) {
      return null
    }

    const mark = ledger.usageStatsOffer

    return mark ? PLACE_BY_MARK[mark] : 'card'
  }
)

/** The card's ×: the question moves to the tray, unanswered. */
export function deferUsageStatsOffer(): void {
  markGuidance('usageStatsOffer', 'deferred')
}

/** The tray item's ×: the person asked not to be offered it here again. */
export function clearUsageStatsOffer(): void {
  markGuidance('usageStatsOffer', 'cleared')
}

/** "Details" on the card and the tray item's action: the short explainer, which can answer. */
export function openUsageStatsDetails(): void {
  $sharedMetricsDetailsOpen.set(true)
}
