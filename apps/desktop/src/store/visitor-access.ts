/**
 * The app's one answer to "must this visitor sign up before anything runs?" and its one way to start
 * sign up (lib/visitor-access.ts has the rule and what is real today).
 *
 * Read once per page: the serving page's globals do not change under it, and signing up is a full
 * navigation away and back. Tests and the dev harness put the app in the signed-out state the way the
 * hosting will, by injecting `__HERMES_SIGN_UP_URL__` before the app loads (or by setting
 * `$visitorAccess` in a unit test).
 */

import { atom, computed } from 'nanostores'

import { isBrowserHostedDesktop } from '@/lib/platform'
import type { BrowserHostGlobals } from '@/lib/platform'
import {
  OPEN_ACCESS,
  signInFailedAt,
  signUpHref,
  type VisitorAccess,
  visitorAccessFrom,
  withoutSignInOutcome
} from '@/lib/visitor-access'

import { discoveryLabVisitorAccess } from './discovery-lab'
import { recordIaAction } from './ia-metrics'
import { $portalShell } from './shell-variant'

export const $visitorAccess = atom<VisitorAccess>(OPEN_ACCESS)

/** Nothing this visitor sends runs until they sign up. Only the redesign holds a message and shows
 *  where it went (the held exchange, the tray card, the splash); classic has no such surface, so it
 *  sends as it always has. */
export const $signUpRequired = computed(
  [$visitorAccess, $portalShell],
  (access, portal) => portal && access.kind === 'sign_up_required'
)

/** Sign up came back without signing the visitor in: the recovery screen is up. Cleared by the next
 *  attempt, or when the visitor puts that screen away. */
export const $signInFailed = atom(false)

/** The visitor put "Let's try signing in again" away (a nav click, Escape): back to the ordinary
 *  signed-out app, their held message still held. */
export function dismissSignInRecovery(): void {
  $signInFailed.set(false)
}

/** main.tsx, before the first render: settle access from the page, and take a failed sign-in's report
 *  off the address so a reload does not say it again. */
export function installVisitorAccess(): void {
  const pageUrl = window.location.href

  // TESTING ONLY: null unless the discovery lab turned its sign-up stand-in on in this tab.
  const access =
    discoveryLabVisitorAccess(pageUrl) ??
    visitorAccessFrom({
      nativeApp: Boolean(window.hermesDesktop) && !isBrowserHostedDesktop(),
      pageUrl,
      signUpUrl: (window as Window & BrowserHostGlobals).__HERMES_SIGN_UP_URL__
    })

  $visitorAccess.set(access)
  $signInFailed.set(access.kind === 'sign_up_required' && signInFailedAt(pageUrl))

  const clean = withoutSignInOutcome(pageUrl)

  if (clean !== pageUrl) {
    window.history.replaceState(window.history.state, '', clean)
  }
}

export type SignUpEntry = 'chat' | 'recovery' | 'sidebar' | 'splash' | 'tray'

/** Leave for sign up, which returns to this same place (this chat, this page) when it is done. A
 *  no-op where sign up is not required. */
export function startSignUp(from: SignUpEntry): void {
  const access = $visitorAccess.get()

  if (access.kind !== 'sign_up_required') {
    return
  }

  recordIaAction(`ia.sign_up.${from}`)
  $signInFailed.set(false)
  window.location.assign(signUpHref(access.signUpUrl, window.location.href))
}
