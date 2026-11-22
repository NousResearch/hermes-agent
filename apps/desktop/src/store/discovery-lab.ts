/**
 * TESTING ONLY (branch `test/discovery-lab`, never merged): the stand-ins the discovery lab
 * (app/discovery-lab) puts in place of facts a test machine cannot produce on demand. Each one is read
 * by exactly one real derivation, and with the lab untouched each is absent, so that derivation
 * answers exactly as it does without this module:
 *
 * - Sign up: `discoveryLabVisitorAccess` (read by `installVisitorAccess`) answers "sign up required"
 *   the way a hosting that injects `__HERMES_SIGN_UP_URL__` would, with the sign-up service replaced
 *   by the lab's own stand-in page. Sign up is a full navigation away and back, which memory does not
 *   survive, so this one lives in the TAB (sessionStorage): gone when the tab closes or the lab turns
 *   it off, never in another tab, never on the machine.
 * - Usage stats: `$discoveryLabUsageStatsConsent` (read by `$sharedMetricsOfferConsent`) stands in for
 *   the backend's "not answered yet". In memory only; an answer given while it stands in retires it
 *   without writing the profile's config.
 */

import type { SharedMetricsConsentResult } from '@hermes/shared'
import { atom } from 'nanostores'

import { SIGN_IN_OUTCOME_PARAM, type VisitorAccess } from '@/lib/visitor-access'

const SIGN_UP_STAND_IN_KEY = 'hermes.desktop.discoveryLab.signUp'

/** The lab's stand-in for the sign-up service: the lab page, in its sign-up mode. */
export const DISCOVERY_LAB_SIGN_UP_ROUTE = '/discovery-lab?stand-in=sign-up'

/** Where sign up says to come back to (lib/visitor-access.ts `signUpHref`). */
export const SIGN_UP_RETURN_PARAM = 'return_to'

/** The sign-up stand-in is on in this tab. */
export function discoveryLabSignUpOn(): boolean {
  try {
    return window.sessionStorage.getItem(SIGN_UP_STAND_IN_KEY) === 'on'
  } catch {
    return false
  }
}

/** Turn the sign-up stand-in on or off for this tab. Takes effect on the next page load, as the
 *  injected URL it stands in for would. */
export function setDiscoveryLabSignUp(on: boolean): void {
  try {
    if (on) {
      window.sessionStorage.setItem(SIGN_UP_STAND_IN_KEY, 'on')
    } else {
      window.sessionStorage.removeItem(SIGN_UP_STAND_IN_KEY)
    }
  } catch {
    // Storage blocked: the lab says the stand-in is off, which is what the next load will find.
  }
}

/** The stand-in sign-up page for the app at `pageUrl`: the same page, at the lab's sign-up route,
 *  with nothing an earlier sign up left on it. */
export function discoveryLabSignUpUrl(pageUrl: string): string {
  const url = new URL(pageUrl)
  url.searchParams.delete(SIGN_IN_OUTCOME_PARAM)
  url.searchParams.delete(SIGN_UP_RETURN_PARAM)
  url.hash = DISCOVERY_LAB_SIGN_UP_ROUTE

  return url.href
}

/** Sign up required with the lab's stand-in as the sign-up service, or null (the page decides). */
export function discoveryLabVisitorAccess(pageUrl: string): null | VisitorAccess {
  return discoveryLabSignUpOn() ? { kind: 'sign_up_required', signUpUrl: discoveryLabSignUpUrl(pageUrl) } : null
}

/** The backend's shared-metrics answer while the lab stands in for it; null leaves the backend's. */
export const $discoveryLabUsageStatsConsent = atom<null | SharedMetricsConsentResult>(null)
