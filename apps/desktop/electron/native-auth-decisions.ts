/**
 * native-auth-decisions.ts
 *
 * Pure decision helpers extracted from main.ts for the RFC 8252 native-app
 * auth flow. These encode three choices that were each the site of a real
 * runtime bug — invisible to the mocked flow tests because the tests never
 * exercised the real main.ts internals. Keeping them pure + unit-tested here
 * prevents silent regressions:
 *
 *   1. resolveJsonBody      — the token/refresh POST body must be the raw
 *      object (fetchJson owns JSON.stringify). Pre-stringifying double-encodes
 *      it into a JSON string, which the gateway's Pydantic model rejects with
 *      422 "Input should be a valid dictionary".
 *
 *   2. oauthSessionIsLive   — an OAuth gateway is "signed in" when EITHER a
 *      native bearer token OR a live cookie session exists. Gating on the
 *      cookie alone rejects a completed native login and loops the UI into
 *      "not signed in".
 *
 *   3. resolveOauthRestAuth — an oauth-mode REST call authenticates with the
 *      native bearer when present, else the cookie partition. Cookie-only
 *      routing returns 401 no_cookie for a cookieless native session.
 *
 * All three are trivial once named; the value is the test that pins the
 * contract so the god-file call sites can't drift back to the buggy shape.
 */

/**
 * Decide the request body to hand to fetchJson (which JSON.stringifies it).
 * Returns the object UNCHANGED — callers must NOT pre-stringify. A string here
 * would be double-encoded downstream; this function exists to document and
 * pin that contract at the one seam that got it wrong.
 */
export function resolveJsonBody<T>(body: T): T {
  return body
}

/**
 * True when an oauth gateway should be treated as signed-in. `hasNativeToken`
 * is whether a native bearer token is stored; `hasCookieSession` is whether a
 * live AT-or-RT cookie exists in the OAuth partition. Either suffices.
 */
export function oauthSessionIsLive(hasNativeToken: boolean, hasCookieSession: boolean): boolean {
  return hasNativeToken || hasCookieSession
}

export type OauthRestAuth =
  | { kind: 'bearer'; token: string }
  | { kind: 'cookie' }

/**
 * Decide how an oauth-mode REST request authenticates: prefer the native
 * bearer (cookieless RFC 8252 flow) when a non-empty access token is present,
 * otherwise fall back to the cookie partition. `nativeAccessToken` is the
 * result of ensureNativeAccessToken (null/empty when there is no native
 * session or the refresh terminally failed).
 */
export function resolveOauthRestAuth(nativeAccessToken: string | null | undefined): OauthRestAuth {
  if (nativeAccessToken) {
    return { kind: 'bearer', token: nativeAccessToken }
  }

  return { kind: 'cookie' }
}
