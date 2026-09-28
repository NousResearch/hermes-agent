import { atom } from 'nanostores'

import { onboardingSurfaceActive } from '@/store/onboarding-presence'
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

/**
 * Why the free tier is not set up, when the backend says it tried and could
 * not. `null` when there is an identity, when the tier is off, or when the
 * backend never reported a failure (an older backend, or no boot yet).
 *
 * `door` is which way forward the copy may honestly offer. The account service
 * that refused is the same one a sign-in goes through: when it is unreachable
 * or erroring, offering sign-in walks the user into a second failure, so those
 * codes get "try again / another provider" only.
 */
export interface FreeTierSetupFailure {
  /** One of the backend's `anon_*` codes (`hermes_cli/anon_auth.py`), or a newer one this build does not know. */
  code: string
  door: 'retry' | 'sign_in'
  message: string
  retryAfter: number
  retryable: boolean
}

const UNREACHABLE_CODES = new Set<string>(['anon_server_error', 'anon_unreachable'])

export function freeTierSetupFailure(status: FreeTierStatus | null): FreeTierSetupFailure | null {
  if (!status || !status.enabled || status.has_guest) {
    return null
  }

  const code = typeof status.error_code === 'string' ? status.error_code.trim() : ''

  if (!code) {
    return null
  }

  return {
    code,
    door: UNREACHABLE_CODES.has(code) ? 'retry' : 'sign_in',
    message: typeof status.error === 'string' ? status.error : '',
    retryAfter: Math.max(0, Math.round(Number(status.retry_after) || 0)),
    retryable: status.retryable === true
  }
}

/**
 * A rounded, spoken duration for copy — "a few seconds", "about a minute",
 * "about 5 minutes", "about an hour" — mirroring the backend's `friendly_wait`
 * so a wait reads the same whichever side phrased it. Never a raw second count.
 */
export function friendlyWait(seconds: number): string {
  const s = Math.max(0, Number.isFinite(seconds) ? seconds : 0)

  if (s <= 15) {
    return 'a few seconds'
  }

  if (s < 90) {
    return 'about a minute'
  }

  if (s < 3600) {
    return `about ${Math.round(s / 60)} minutes`
  }

  const hours = Math.round(s / 3600)

  return hours <= 1 ? 'about an hour' : `about ${hours} hours`
}

/**
 * The user's own retry of the free-tier set-up (`free_tier.provision`): the one
 * attempt that may run inside the backend's cooldown. Re-reads the status
 * afterwards so every surface keyed on it moves together. Returns the fresh
 * status, or the last known one when the call itself failed.
 */
export async function provisionFreeTier(requestGateway: FreeTierRequester): Promise<FreeTierStatus | null> {
  try {
    await requestGateway('free_tier.provision')
  } catch {
    // The status read below still reports what the backend knows.
  }

  return refreshFreeTierStatus(requestGateway)
}

/** Persist the one-time notice acknowledgement, then re-read so every surface
 *  keyed on `notice_pending` drops away together. */
export async function ackFreeTierNotice(requestGateway: FreeTierRequester): Promise<boolean> {
  try {
    const result = await requestGateway<{ acked?: boolean }>('free_tier.ack_notice')

    if (result?.acked !== true) {
      return false
    }
  } catch {
    // A failed ack means the notice is still owed; the caller keeps its surface up.
    return false
  }

  await refreshFreeTierStatus(requestGateway)

  return true
}

/**
 * Whether the SELECTED route runs on the free tier: `setup.runtime_check.free_tier`, keyed on the
 * endpoint the backend resolved, not on profile state. `null` until a readiness round answers. A
 * free-tier identity beside the user's own key reads `false` here while `$freeTierStatus.available`
 * stays true — that split is what picks the intro's shape.
 */
export const $freeTierRoute = atom<boolean | null>(null)

export function setFreeTierRoute(route: boolean | null | undefined) {
  $freeTierRoute.set(typeof route === 'boolean' ? route : null)
}

/** True when the one-time introduction is still owed to this user. */
export function freeTierNoticePending(status: FreeTierStatus | null): boolean {
  return Boolean(status?.has_guest && status.notice_pending)
}

/** The introduction is owed AND the free tier is the route: the overlay's ready screen. */
export function freeTierReadyPending(status: FreeTierStatus | null, route: boolean | null): boolean {
  return freeTierNoticePending(status) && route === true
}

/**
 * True when the introduction is owed AND a provider of the user's own carries
 * inference — the case the composer strip covers. When the free tier itself
 * carries inference the onboarding overlay's ready screen owns the moment
 * instead, so the two can never both be on screen.
 */
export function freeTierStripPending(status: FreeTierStatus | null, route: boolean | null): boolean {
  return freeTierNoticePending(status) && route === false && !onboardingSurfaceActive()
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
