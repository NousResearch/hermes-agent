/**
 * Who may run anything here: the one question behind "nothing runs until they sign up" (web app
 * build plan, Basics). Pure: the store (`store/visitor-access.ts`) feeds it the page's facts once.
 *
 * What is real today: every page that renders this app is served by a Hermes backend the visitor is
 * already allowed to use (native Desktop on its own machine, or a Webapp a machine serves behind its
 * own sign-in or launch link), so the answer is always `open`. A page served to someone with no
 * account does not exist yet; the hosting that will serve it says so by injecting
 * `__HERMES_SIGN_UP_URL__` (where sign up starts) the way a Hermes server injects its other
 * `__HERMES_*` globals. The native app never reads it: Hermes on your own computer never asks you to
 * sign up to chat.
 */

/** `open`: messages run. `sign_up_required`: they are held until the visitor signs up at `signUpUrl`. */
export type VisitorAccess = { kind: 'open' } | { kind: 'sign_up_required'; signUpUrl: string }

export const OPEN_ACCESS: VisitorAccess = { kind: 'open' }

export interface VisitorAccessEvidence {
  /** The renderer is the native app, not a page a server sent to a browser. */
  nativeApp: boolean
  /** The serving page's `__HERMES_SIGN_UP_URL__`, as injected (untyped until checked here). */
  signUpUrl: unknown
  /** This page's own URL: a relative sign-up URL resolves against it. */
  pageUrl: string
}

function webUrl(value: unknown, base: string): null | URL {
  if (typeof value !== 'string' || !value.trim()) {
    return null
  }

  try {
    const url = new URL(value.trim(), base)

    return url.protocol === 'https:' || url.protocol === 'http:' ? url : null
  } catch {
    return null
  }
}

export function visitorAccessFrom({ nativeApp, pageUrl, signUpUrl }: VisitorAccessEvidence): VisitorAccess {
  const url = nativeApp ? null : webUrl(signUpUrl, pageUrl)

  return url ? { kind: 'sign_up_required', signUpUrl: url.href } : OPEN_ACCESS
}

/**
 * Where sign up comes back to when it could not sign the visitor in (build plan flow 2): the page
 * it was started from, carrying this parameter. Agreed with the portal's hand-off, which does not
 * send it yet.
 */
export const SIGN_IN_OUTCOME_PARAM = 'hermes_sign_in'

/** The URL that starts sign up and brings the visitor back to `returnTo` afterwards. */
export function signUpHref(signUpUrl: string, returnTo: string): string {
  const url = new URL(signUpUrl)
  url.searchParams.set('return_to', withoutSignInOutcome(returnTo))

  return url.href
}

/** Whether `pageUrl` is sign up coming back after it failed to sign the visitor in. */
export function signInFailedAt(pageUrl: string): boolean {
  return new URL(pageUrl).searchParams.get(SIGN_IN_OUTCOME_PARAM) === 'failed'
}

/** `pageUrl` without the sign-in outcome, so a reload or a second try does not report it again. */
export function withoutSignInOutcome(pageUrl: string): string {
  const url = new URL(pageUrl)
  url.searchParams.delete(SIGN_IN_OUTCOME_PARAM)

  return url.href
}
