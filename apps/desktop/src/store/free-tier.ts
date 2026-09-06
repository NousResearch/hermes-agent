import { atom } from 'nanostores'

import type { FreeTierStatus } from '@/types/hermes'

/** The model the free-tier route runs on. Used to recognise a session that is
 *  still homed on the free tier after a sign-in. */
export const FREE_TIER_MODEL = 'nous/welcome'

/** The provider slug the free-tier route and a signed-in Nous account share. */
export const NOUS_PROVIDER_ID = 'nous'

export type FreeTierRequester = <T = unknown>(method: string, params?: Record<string, unknown>) => Promise<T>

/**
 * The backend's free-tier verdict, cached for the chrome that paints it (the
 * statusbar chip, the first-launch intro, the billing view). The backend is
 * authoritative — this atom is only a cache of `free_tier.status`, which is a
 * local, zero-network read — so nothing here ever decides on its own that the
 * free tier is on or off. `null` means "not asked yet"; every consumer must
 * render as if there were no free tier until an answer lands.
 */
export const $freeTierStatus = atom<FreeTierStatus | null>(null)

function isFreeTierStatus(value: unknown): value is FreeTierStatus {
  return typeof value === 'object' && value !== null && typeof (value as FreeTierStatus).has_guest === 'boolean'
}

/**
 * Pull the current status. No polling loop of its own: callers ride an existing
 * cadence (the ambient status snapshot) or a seam that just changed the answer
 * (boot, a completed sign-in, an acknowledged notice).
 *
 * A failed read leaves the last known answer in place rather than blanking the
 * chrome — an older backend without the method, or a gateway flap, is not
 * evidence that the free tier went away.
 */
export async function refreshFreeTierStatus(requestGateway: FreeTierRequester): Promise<FreeTierStatus | null> {
  try {
    const status = await requestGateway<FreeTierStatus>('free_tier.status')

    if (!isFreeTierStatus(status)) {
      return $freeTierStatus.get()
    }

    $freeTierStatus.set(status)

    return status
  } catch {
    return $freeTierStatus.get()
  }
}

/** Persist the one-time notice acknowledgement, then re-read so every surface
 *  keyed on `notice_pending` drops away together. */
export async function ackFreeTierNotice(requestGateway: FreeTierRequester): Promise<void> {
  try {
    await requestGateway('free_tier.ack_notice')
  } catch {
    // The notice is cosmetic; a failed ack just means it is offered again.
    return
  }

  await refreshFreeTierStatus(requestGateway)
}

/** True when the one-time introduction is still owed to this user. */
export function freeTierNoticePending(status: FreeTierStatus | null): boolean {
  return Boolean(status?.has_guest && status.notice_pending)
}

/**
 * True when the introduction is owed AND a provider of the user's own carries
 * inference — the case the composer strip covers. When the free tier itself
 * carries inference the onboarding overlay's ready screen owns the moment
 * instead, so the two can never both be on screen.
 */
export function freeTierStripPending(status: FreeTierStatus | null): boolean {
  return freeTierNoticePending(status) && !status?.carries_inference
}

// Several composers can be mounted at once (split zones, a popout mid-dock).
// The FIRST mounted strip claims the notice; the rest render nothing, so one
// pending notice never paints N times. Mirrors the real-profile-consent claim.
const $noticeClaim = atom<null | string>(null)

export function claimFreeTierNotice(id: string) {
  if ($noticeClaim.get() === null) {
    $noticeClaim.set(id)
  }
}

export function releaseFreeTierNotice(id: string) {
  if ($noticeClaim.get() === id) {
    $noticeClaim.set(null)
  }
}

export function freeTierNoticeClaim() {
  return $noticeClaim
}
