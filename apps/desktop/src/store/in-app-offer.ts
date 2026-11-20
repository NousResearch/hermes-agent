import { atom } from 'nanostores'

import { persistString, persistStringArray, storedString, storedStringArray } from '@/lib/storage'
import { $composerDraft } from '@/store/composer'
import { $freeTierStatus, type FreeTierRequester } from '@/store/free-tier'
import { $freeTierSignIn } from '@/store/free-tier-sign-in'
import { $onboardingSurfaceClear } from '@/store/onboarding-presence'
import { $busy } from '@/store/session'
import type { InAppOfferPayload } from '@/types/hermes'

/**
 * The Nous Portal's in-app offer, renderer half. The backend memoises the
 * Portal's campaign answer and serves it on `free_tier.status`
 * (`in_app_offer`); this store decides whether, and when, it becomes a
 * dialog.
 *
 * Presentation rules, each one guarding a way the dialog could misfire:
 *
 * - Once per offer. Dismissal is permanent per offer id and lives here, not on
 *   the backend: what the user has seen is a presentation fact.
 * - Never mid-work. The dialog waits for an idle moment: no turn running, an
 *   empty composer, a visible and focused window. A dialog that steals focus
 *   while someone types gets dismissed by the next keystroke, unread.
 * - Never over another first-run or sign-in surface.
 * - One window. Only the window holding the claim shows it, so two windows
 *   never stack two dialogs for one offer.
 * - At most one offer dialog per day, across campaigns.
 *
 * The backend memo only fills on an account fetch, so this store also asks
 * for one on a slow cadence (`in_app_offer.refresh`); see
 * `startInAppOfferRefresh`.
 */

/** Offers the user closed. Dismissal is permanent per offer id. */
const DISMISSED_KEY = 'hermes.in-app-offer.dismissed-v1'
/** When the last offer dialog was shown (epoch ms). Caps frequency. */
const LAST_SHOWN_KEY = 'hermes.in-app-offer.last-shown-v1'
/** At most one offer dialog per day, whatever the campaign cadence. */
const RESHOW_QUIET_MS = 24 * 60 * 60 * 1000
/** How long the app must sit idle before the dialog may open. Long enough
 * that a pause between two sentences does not count as idle. */
export const IDLE_SETTLE_MS = 8_000
/** The account-fetch cadence behind `in_app_offer.refresh`. Campaigns are
 * slow-moving; one cheap read per half hour is plenty. */
export const REFRESH_EVERY_MS = 30 * 60 * 1000
/** The first fetch waits for boot to settle so it never competes with the
 * gateway's own startup reads. */
export const FIRST_REFRESH_DELAY_MS = 20_000

/** The offer the dialog should show right now, or null. */
export const $inAppOfferDialog = atom<InAppOfferPayload | null>(null)

const $dismissedIds = atom<string[]>(storedStringArray(DISMISSED_KEY))

/** Which mounted host owns the dialog (the sign-in dialog's claim pattern). */
const $claim = atom<null | string>(null)

export function claimInAppOffer(id: string): void {
  if ($claim.get() === null) {
    $claim.set(id)
  }
}

export function releaseInAppOffer(id: string): void {
  if ($claim.get() === id) {
    $claim.set(null)
    $inAppOfferDialog.set(null)
  }
}

export function inAppOfferClaim() {
  return $claim
}

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
 * still serves it. Re-reads storage first so a dismissal made in another
 * window (same localStorage) is kept rather than overwritten. */
export function dismissInAppOffer(id: string): void {
  const stored = storedStringArray(DISMISSED_KEY)
  const next = stored.includes(id) ? stored : [...stored, id]

  persistStringArray(DISMISSED_KEY, next)
  $dismissedIds.set(next)
  $inAppOfferDialog.set(null)
}

/** True when nothing the user is doing would be interrupted by a dialog. */
export function appIsIdle(): boolean {
  if ($busy.get() || $composerDraft.get().trim().length > 0) {
    return false
  }

  if (typeof document !== 'undefined' && (document.visibilityState === 'hidden' || !document.hasFocus())) {
    return false
  }

  return true
}

/** No other first-run or sign-in surface owns the screen. */
function screenIsFree(): boolean {
  return $onboardingSurfaceClear.get() && $freeTierSignIn.get().status === 'closed'
}

/**
 * The offer the status carries if it may be shown at all, ignoring timing:
 * valid, not dismissed (in this window or another), and past the frequency
 * cap. Null otherwise.
 */
export function eligibleOffer(status: unknown): InAppOfferPayload | null {
  const offer = offerFromStatus(status)

  if (!offer) {
    return null
  }

  // Another window may have dismissed it: storage is the shared truth.
  if (storedStringArray(DISMISSED_KEY).includes(offer.id) || $dismissedIds.get().includes(offer.id)) {
    return null
  }

  if (Date.now() - lastShownAt() < RESHOW_QUIET_MS) {
    return null
  }

  return offer
}

/**
 * Decide, right now, whether the dialog opens. Pure apart from the
 * (idempotent) dialog set, so re-running it on every signal is free. `idle`
 * is the caller's settled-idle verdict; the dialog only ever OPENS on an idle
 * moment, but once open it stays until the user closes it.
 */
export function syncInAppOffer(status: unknown, { idle }: { idle: boolean }): void {
  const current = $inAppOfferDialog.get()

  if (current) {
    // Already showing: keep it until the user acts, unless the Portal
    // withdrew it (campaign ended, account changed or signed out).
    if (offerFromStatus(status)?.id !== current.id) {
      $inAppOfferDialog.set(null)
    }

    return
  }

  const offer = eligibleOffer(status)

  if (!offer || !idle || !screenIsFree()) {
    return
  }

  $inAppOfferDialog.set(offer)
}

/**
 * Wire the ambient decision for the owning window: every free-tier status
 * read, onboarding or sign-in leaving the screen, and the app settling into
 * an idle moment re-run it. Returns a stop function.
 */
export function listenInAppOfferStatus(): () => void {
  let idleTimer: null | ReturnType<typeof setTimeout> = null

  const clearIdle = () => {
    if (idleTimer !== null) {
      clearTimeout(idleTimer)
      idleTimer = null
    }
  }

  // Each signal restarts the settle clock; the decision runs with idle=true
  // only after IDLE_SETTLE_MS of continuous quiet.
  const arm = () => {
    clearIdle()
    syncInAppOffer($freeTierStatus.get(), { idle: false })

    if (!eligibleOffer($freeTierStatus.get()) || !appIsIdle()) {
      return
    }

    idleTimer = setTimeout(() => {
      idleTimer = null
      syncInAppOffer($freeTierStatus.get(), { idle: appIsIdle() })
    }, IDLE_SETTLE_MS)
  }

  const stops = [
    $freeTierStatus.listen(arm),
    $onboardingSurfaceClear.listen(arm),
    $freeTierSignIn.listen(arm),
    $busy.listen(arm),
    $composerDraft.listen(arm)
  ]

  const onWindowSignal = () => arm()

  if (typeof window !== 'undefined') {
    window.addEventListener('focus', onWindowSignal)
    window.addEventListener('blur', onWindowSignal)
    document.addEventListener('visibilitychange', onWindowSignal)
  }

  arm()

  return () => {
    clearIdle()
    stops.forEach(stop => stop())

    if (typeof window !== 'undefined') {
      window.removeEventListener('focus', onWindowSignal)
      window.removeEventListener('blur', onWindowSignal)
      document.removeEventListener('visibilitychange', onWindowSignal)
    }
  }
}

/**
 * Ask the backend for a fresh account read on a slow cadence and fold the
 * answer into the free-tier status the decision above reads. Without this the
 * offer only arrives when some other feature happens to fetch the account,
 * which for a user on another provider may be never. Failures are silent: the
 * offer is a nice-to-have and must never surface an error.
 */
export function startInAppOfferRefresh(requestGateway: FreeTierRequester): () => void {
  let stopped = false
  let timer: null | ReturnType<typeof setTimeout> = null

  const run = async () => {
    try {
      const result = await requestGateway<{ in_app_offer?: InAppOfferPayload | null }>('in_app_offer.refresh')

      if (stopped) {
        return
      }

      const status = $freeTierStatus.get()

      if (status) {
        // Same atom the status poll writes, so one decision path reads both.
        $freeTierStatus.set({ ...status, in_app_offer: result?.in_app_offer ?? null })
      }
    } catch {
      // Older backend without the method, or a transient failure: stay quiet.
    }

    if (!stopped) {
      timer = setTimeout(run, REFRESH_EVERY_MS)
    }
  }

  timer = setTimeout(run, FIRST_REFRESH_DELAY_MS)

  return () => {
    stopped = true

    if (timer !== null) {
      clearTimeout(timer)
    }
  }
}

/** Offer events this install already reported (`<id>:shown`, `<id>:clicked`). */
const REPORTED_KEY = 'hermes.in-app-offer.reported-v1'

export type InAppOfferEvent = 'clicked' | 'shown'

/**
 * Tell the Portal the offer was shown or clicked, once per offer per event
 * on this install (the Portal's campaign counts). The record is written
 * before the call, so a second window or a retry never double-counts; a
 * report that fails is not retried, since a count is not worth a second
 * request. Silent on any failure, including an older backend without the
 * method.
 */
export function reportInAppOfferEvent(
  requestGateway: FreeTierRequester,
  offerId: string,
  event: InAppOfferEvent
): void {
  const key = `${offerId}:${event}`
  const reported = storedStringArray(REPORTED_KEY)

  if (reported.includes(key)) {
    return
  }

  // Bounded like the dismissed list's growth: keep the newest entries.
  persistStringArray(REPORTED_KEY, [...reported, key].slice(-100))
  void requestGateway('in_app_offer.report', { event, offer_id: offerId }).catch(() => undefined)
}

/** Test seam: reset module state between cases. */
export function resetInAppOfferStateForTest(): void {
  $inAppOfferDialog.set(null)
  $claim.set(null)
  $dismissedIds.set(storedStringArray(DISMISSED_KEY))
  persistString(LAST_SHOWN_KEY, null)
  persistStringArray(REPORTED_KEY, [])
}
