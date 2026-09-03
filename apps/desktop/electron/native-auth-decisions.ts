/**
 * native-auth-decisions.ts
 *
 * Pure decision helpers extracted from main.ts for the RFC 8252 native-app
 * auth flow. These encode six choices that were each the site of a real
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
 *   4. resolveReadinessProbeAuth — the boot readiness probe must authenticate
 *      the same way the rest of the connection does. A credential-free probe
 *      against a gated gateway 401s forever; worse, it cannot tell a missing
 *      route from a rejected session (see backend-health.ts).
 *
 *   5. oauthGuardMayHardFail — `auth_required: true` means "this gateway is
 *      gated", NOT "this gateway speaks OAuth". A password-provider gateway
 *      can satisfy neither the native-bearer nor the OAuth-partition-cookie
 *      check by design, so the pre-flight guard must not hard-fail it.
 *
 *   6. resolveGatedDownloadAuth — file save/read must present the SAME
 *      credentials as oauth REST. `saveGatewayFile` used to always ride the
 *      OAuth cookie partition, so a cookieless native (or native-password)
 *      session could list files via `hermes:api` and still 401 on Download.
 *
 * All six are trivial once named; the value is the test that pins the
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

export type OauthRestAuth = { kind: 'bearer'; token: string } | { kind: 'cookie' }

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

export type ReadinessProbeAuth = OauthRestAuth | { kind: 'token'; token: string | null } | { kind: 'public' }

/**
 * Decide how the boot readiness probe authenticates.
 *
 * The probe must present the SAME credentials the rest of the connection
 * will use. A credential-free probe against a gated gateway 401s until the
 * boot deadline even though the session is perfectly valid — and because the
 * dashboard auth gate runs ahead of the SPA catch-all, an unknown `/api/*`
 * path answers 401 rather than 404, so the probe also cannot detect a backend
 * that predates `/api/health`. Sending credentials is what lets a missing
 * route surface as a real 404 (see `isMissingHealthEndpointError`).
 *
 * `oauth` reuses `resolveOauthRestAuth` so the probe and every other oauth
 * REST call make the identical bearer-vs-cookie choice. `token` presents the
 * connection's session token. `local` (and anything unrecognized) stays
 * public: a loopback backend has no gate, and sending credentials it never
 * issued would be meaningless.
 */
export function resolveReadinessProbeAuth(
  authMode: string | null | undefined,
  nativeAccessToken?: string | null,
  connectionToken?: string | null
): ReadinessProbeAuth {
  if (authMode === 'oauth') {
    return resolveOauthRestAuth(nativeAccessToken)
  }

  if (authMode === 'token') {
    return { kind: 'token', token: connectionToken ?? null }
  }

  return { kind: 'public' }
}

export type GatedDownloadAuth = OauthRestAuth | { kind: 'token'; token: string | null }

/**
 * Decide how a gated file download authenticates.
 *
 * Must match oauth REST (`resolveOauthRestAuth`): native bearer when present,
 * else the OAuth cookie partition. Token/local connections keep the static
 * session-token header. A cookie-only download against a cookieless native
 * session is the #88987 401 — Files panel listing works, Download does not.
 */
export function resolveGatedDownloadAuth(
  authMode: string | null | undefined,
  nativeAccessToken?: string | null,
  connectionToken?: string | null
): GatedDownloadAuth {
  if (authMode === 'oauth') {
    return resolveOauthRestAuth(nativeAccessToken)
  }

  return { kind: 'token', token: connectionToken ?? null }
}

export interface AdvertisedAuthProvider {
  name?: string
  supportsPassword?: boolean
}

/** Dashboard `basic` auth is username/password; `/api/status` often lists it as a bare string. */
const PASSWORD_PROVIDER_NAMES = new Set(['basic'])

const OAUTH_NOT_SIGNED_IN_MESSAGE =
  'Remote Hermes gateway uses OAuth, but you are not signed in. ' +
  'Open Settings → Gateway and click "Sign in", or switch back to Local.'

const OAUTH_SESSION_EXPIRED_MESSAGE =
  'Your remote gateway session has expired. Open Settings → Gateway and click "Sign in" again.'

/**
 * Normalize `/api/auth/providers` objects *or* `/api/status` `auth_providers`
 * string names into the shape `oauthGuardMayHardFail` understands.
 */
export function normalizeAdvertisedAuthProviders(providers: unknown): AdvertisedAuthProvider[] {
  if (!Array.isArray(providers)) {
    return []
  }

  const out: AdvertisedAuthProvider[] = []

  for (const provider of providers) {
    if (typeof provider === 'string') {
      const name = provider.trim()

      if (!name) {
        continue
      }

      out.push({ name, supportsPassword: PASSWORD_PROVIDER_NAMES.has(name) })

      continue
    }

    if (!provider || typeof provider !== 'object') {
      continue
    }

    const raw = provider as AdvertisedAuthProvider & { supports_password?: boolean }
    const name = typeof raw.name === 'string' ? raw.name.trim() : ''

    if (!name) {
      continue
    }

    const supportsPassword =
      typeof raw.supportsPassword === 'boolean'
        ? raw.supportsPassword
        : typeof raw.supports_password === 'boolean'
          ? raw.supports_password
          : PASSWORD_PROVIDER_NAMES.has(name)

    out.push({ name, supportsPassword })
  }

  return out
}

/**
 * A 401/403 on `POST /api/auth/ws-ticket` is "session expired" only when we
 * actually had a decryptable native token set. Stale partition cookies plus
 * an unreadable keychain otherwise look like a live oauth session and the
 * ticket mint 401s — that must send the user to Sign in, not "expired".
 */
export function oauthTicketFailureAuthMessage(hasDecryptableNativeSession: boolean): string {
  return hasDecryptableNativeSession ? OAUTH_SESSION_EXPIRED_MESSAGE : OAUTH_NOT_SIGNED_IN_MESSAGE
}

/**
 * Whether the oauth pre-flight guard may hard-fail a connection for "not
 * signed in".
 *
 * `authModeFromStatus` maps the gateway's `auth_required: true` onto
 * `'oauth'`, but that flag only means the dashboard is GATED — it says
 * nothing about how you authenticate. A gateway whose providers are all
 * username/password cannot satisfy the guard's checks by construction:
 * `start_login` raises NotImplementedError, `/auth/native/authorize` rejects
 * password providers, and its cookies are set by a plain password-login POST
 * rather than the `/auth/callback` redirect the OAuth partition is primed
 * for. Hard-failing there rejects a live session one line before the
 * ws-ticket mint that would have succeeded against that very partition.
 *
 * Returns false only when EVERY advertised provider is password-based. An
 * unknown or empty list keeps the strict guard, so backends that predate
 * `/api/auth/providers` are unaffected.
 */
export function oauthGuardMayHardFail(providers: unknown): boolean {
  const named = normalizeAdvertisedAuthProviders(providers)

  if (named.length === 0) {
    return true
  }

  return !named.every(provider => provider.supportsPassword)
}
