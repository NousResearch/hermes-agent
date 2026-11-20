import { atom } from 'nanostores'

import { persistString, persistStringArray, storedString, storedStringArray } from '@/lib/storage'
import { $freeTierStatus } from '@/store/free-tier'
import { $onboardingSurfaceClear } from '@/store/onboarding-presence'
import type { InAppOfferPayload } from '@/types/hermes'

/**
 * The Nous Portal's in-app offer, renderer half. The backend memoises the
 * Portal's campaign answer on the account fetch and serves it on
 * `free_tier.status` (`in_app_offer`); this store decides whether it becomes
 * a dialog, exactly once per offer, and never while onboarding owns the
 * screen.
 *
 * The backend holds no dismissal state on purpose: what the user has seen is
 * a presentation fact, so it lives here, keyed by offer id. The Portal can
 * end a campaign any time; the next account fetch drops the field and this
 * store goes quiet on its own.
 */

/** Offers the user closed. Dismissal is permanent per offer id. */
const DISMISSED_KEY = 'hermes.in-app-offer.dismissed-v1'
/** When the last offer dialog was shown (epoch ms). Caps frequency. */
const LAST_SHOWN_KEY = 'hermes.in-app-offer.last-shown-v1'
/** At most one offer dialog per day, whatever the campaign cadence. */
const RESHOW_QUIET_MS = 24 * 60 * 60 * 1000

/** The offer the dialog should show right now, or null. */
export const $inAppOfferDialog = atom<InAppOfferPayload | null>(null)

const $dismissedIds = atom<string[]>(storedStringArray(DISMISSED_KEY))

function lastShownAt(): number {
  const raw = storedString(LAST_SHOWN_KEY)

  return raw ? Number.parseInt(raw, 10) || 0 : 0
}

function offerFromStatus(status: unknown): InAppOfferPayload | null {
  if (typeof status !== 'object' || status === null) {
    return null
  }

  const candidate = (status as { in_app_offer?: unknown }).in_app_offer

  if (typeof candidate !== 'object' || candidate === null) {
    return null
  }

  const offer = candidate as Partial<InAppOfferPayload>

  return typeof offer.id === 'string' &&
    offer.id.length > 0 &&
    typeof offer.title === 'string' &&
    typeof offer.body === 'string' &&
    typeof offer.cta_label === 'string' &&
    typeof offer.cta_url === 'string'
    ? (offer as InAppOfferPayload)
    : null
}

/** Record that this offer was presented, starting the frequency cap. */
export function markInAppOfferShown(): void {
  persistString(LAST_SHOWN_KEY, String(Date.now()))
}

/** Close the offer permanently: it never comes back, even if the campaign
 * still serves it. */
export function dismissInAppOffer(id: string): void {
  const next = $dismissedIds.get().includes(id)
    ? $dismissedIds.get()
    : [...$dismissedIds.get(), id]

  persistStringArray(DISMISSED_KEY, next)
  $dismissedIds.set(next)
  $inAppOfferDialog.set(null)
}

/** Decide whether the latest status carries an offer worth showing. Pure
 * apart from the (idempotent) dialog set, so re-running it on every status
 * tick is free: the same offer does not re-open, a dismissed offer stays
 * closed, and the frequency cap holds new campaigns back for a day. */
export function syncInAppOffer(status: unknown): void {
  const offer = offerFromStatus(status)

  if (!offer) {
    return
  }

  if (!$onboardingSurfaceClear.get()) {
    // Onboarding owns the moment; its clear flips the listener below and the
    // offer re-evaluates then.
    return
  }

  if ($dismissedIds.get().includes(offer.id)) {
    return
  }

  if (Date.now() - lastShownAt() < RESHOW_QUIET_MS) {
    return
  }

  const current = $inAppOfferDialog.get()

  $inAppOfferDialog.set(current?.id === offer.id ? current : offer)
}

/** Wire the ambient refresh: every free-tier status read (the 60s snapshot
 * cadence, sign-ins, boot) re-runs the decision, and onboarding leaving the
 * screen does too. Returns a stop function. */
export function listenInAppOfferStatus(): () => void {
  const sync = () => syncInAppOffer($freeTierStatus.get())

  const stops = [$freeTierStatus.listen(sync), $onboardingSurfaceClear.listen(sync)]

  sync()

  return () => {
    stops.forEach(stop => stop())
  }
}

/** Test seam: reset module state between cases. */
export function resetInAppOfferStateForTest(): void {
  $inAppOfferDialog.set(null)
  $dismissedIds.set(storedStringArray(DISMISSED_KEY))
  persistString(LAST_SHOWN_KEY, null)
}
