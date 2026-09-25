/**
 * portal-session.ts
 *
 * Hermes Desktop's own session with the Nous portal, as the public OAuth
 * client `hermes-desktop` (see portal-oauth.ts for the wire shapes).
 *
 *   - Sign-in opens the user's DEFAULT BROWSER (RFC 8252 loopback + PKCE);
 *     the portal handles login, team choice and consent there. No embedded
 *     window and no partition cookies are involved.
 *   - The token set is persisted through the same encrypted native token
 *     store the gateway flow uses (keyed by the portal base URL), so it obeys
 *     the same keychain-optional storage policy and survives restarts.
 *   - Access tokens renew through the shared native access-token coordinator:
 *     concurrent callers share one refresh flight, the rotated refresh token
 *     is persisted BEFORE the new access token is handed out, and a rejected
 *     refresh (400 invalid_grant) clears the store — i.e. signed out.
 *   - exchangeForAgent() turns the desktop token into a bearer for one Hermes
 *     Cloud agent (RFC 8693, §5). It issues no refresh token; callers renew by
 *     exchanging again.
 *
 * All I/O is injected so the whole lifecycle unit-tests without Electron.
 */

import type http from 'node:http'

import { readStatusCode } from './api-transport'
import {
  CLOUD_NOT_SIGNED_IN_MESSAGE,
  CLOUD_SESSION_EXPIRED_MESSAGE,
  cloudAgentAccessLostError,
  cloudExchangeRateLimitedError,
  cloudLoginRequiredError
} from './cloud-auth-errors'
import {
  createNativeAccessTokenCoordinator,
  type NativeAccessTokenOptions,
  NativeAuthChangedError
} from './native-access-token'
import { NativeLoginCancelledError, type NativeTokenSet, tokenNeedsRefresh } from './native-oauth'
import { runLoopbackAuthorization } from './native-oauth-login'
import {
  agentTokenExchangeGrant,
  authorizationCodeGrant,
  buildPortalAuthorizeUrl,
  isPortalRateLimited,
  oauthErrorCode,
  parseAgentTokenResponse,
  parsePortalTokenResponse,
  portalTokenOrgId,
  portalTokenUrl,
  refreshTokenGrant,
  retryAfterSeconds
} from './portal-oauth'

const TOKEN_REQUEST_TIMEOUT_MS = 15_000

export interface PortalSessionDependencies {
  resolvePortalBaseUrl: () => string
  /** Encrypted native token store, keyed here by the portal base URL. */
  loadTokens: (key: string) => NativeTokenSet | null
  storeTokens: (key: string, tokens: NativeTokenSet) => void
  clearTokens: (key: string) => void
  /** Cookieless JSON POST; throws an httpStatusError on >= 400. */
  postJson: (url: string, body: unknown, opts?: { timeoutMs?: number }) => Promise<any>
  /** shell.openExternal — the user's default browser. */
  openExternal: (url: string) => Promise<void>
  createServer?: typeof http.createServer
  loginTimeoutMs?: number
  nowSeconds?: () => number
  /**
   * Monotonic seconds for the §5 rate-limit backoff, so a wall-clock
   * rollback cannot stretch it. Defaults to `nowSeconds` when that is
   * injected (tests), else `performance.now()`.
   */
  monotonicSeconds?: () => number
  rememberLog?: (message: string) => void
  /**
   * Every successful sign-in reports the org the new desktop token is pinned
   * to (its `org_id` claim, read for this comparison only; null when absent).
   * The agent registry compares it with the org it was populated under — so
   * an org switch is detected even across a sign-out, when no previous token
   * is left to compare against.
   */
  onSignedIn?: (orgId: null | string) => void
}

export interface PortalLoginResult {
  signedIn: boolean
  cancelled?: true
}

// §3 failures that mean the refresh token is dead (revoked, reused, expired,
// or the client itself was refused). Anything else — network, 5xx — keeps the
// token for the next attempt.
function isRefreshRejection(error: unknown): boolean {
  const status = readStatusCode(error)
  const code = oauthErrorCode(error)

  return status === 401 || (status === 400 && (code === 'invalid_grant' || code === 'unauthorized_client'))
}

export function createPortalSession(deps: PortalSessionDependencies) {
  const nowSeconds = () => deps.nowSeconds?.() ?? Math.floor(Date.now() / 1_000)
  const monotonicSeconds = deps.monotonicSeconds ?? (deps.nowSeconds ? nowSeconds : () => performance.now() / 1_000)
  const portal = () => deps.resolvePortalBaseUrl()
  const log = deps.rememberLog ?? (() => undefined)

  const coordinator = createNativeAccessTokenCoordinator({
    clearTokens: deps.clearTokens,
    isRefreshAuthRejection: isRefreshRejection,
    loadTokens: deps.loadTokens,
    normalizeBaseUrl: url => String(url).trim().replace(/\/+$/, ''),
    nowSeconds,
    refreshTokens: async (portalBaseUrl, tokens) =>
      parsePortalTokenResponse(
        await deps.postJson(portalTokenUrl(portalBaseUrl), refreshTokenGrant(tokens.refreshToken), {
          timeoutMs: TOKEN_REQUEST_TIMEOUT_MS
        }),
        nowSeconds()
      ),
    storeTokens: deps.storeTokens,
    tokenNeedsRefresh
  })

  /** A refresh token (or a still-unexpired access token) is a live session. */
  function hasLivePortalSession(): boolean {
    const tokens = deps.loadTokens(portal())

    return Boolean(tokens && (tokens.refreshToken || !tokenNeedsRefresh(tokens, nowSeconds(), 0)))
  }

  /** Current desktop access token, refreshed when near expiry; null = signed out. */
  function getPortalAccessToken(options: NativeAccessTokenOptions = {}): Promise<null | string> {
    return coordinator.ensure(portal(), options)
  }

  // The newest pending browser flow: its cancel handle and the authorize URL
  // (for the "browser didn't open? copy the link" fallback).
  let pending: null | { controller: AbortController; authorizeUrl: null | string } = null

  /** Abort the pending browser sign-in; it resolves as a clean cancel. */
  function cancelLogin(): boolean {
    if (!pending) {
      return false
    }

    pending.controller.abort()

    return true
  }

  function pendingAuthorizeUrl(): null | string {
    return pending?.authorizeUrl ?? null
  }

  async function login(): Promise<PortalLoginResult> {
    const portalBaseUrl = portal()
    const isCurrent = coordinator.beginLogin(portalBaseUrl)
    const flow = { controller: new AbortController(), authorizeUrl: null as null | string }

    // A newer sign-in supersedes the pending one: close its loopback listener
    // now (it resolves as a clean cancel) instead of leaving it open until
    // its timeout with no way left to cancel it.
    pending?.controller.abort()
    pending = flow

    let tokens: NativeTokenSet

    try {
      tokens = await runLoopbackAuthorization(
        {
          openExternal: deps.openExternal,
          createServer: deps.createServer,
          timeoutMs: deps.loginTimeoutMs,
          rememberLog: deps.rememberLog,
          signal: flow.controller.signal,
          onAuthorizeUrl: url => {
            flow.authorizeUrl = url
          }
        },
        {
          buildAuthorizeUrl: params => buildPortalAuthorizeUrl(portalBaseUrl, params),
          redeem: async ({ code, verifier, redirectUri }) =>
            parsePortalTokenResponse(
              await deps.postJson(
                portalTokenUrl(portalBaseUrl),
                authorizationCodeGrant({ code, codeVerifier: verifier, redirectUri }),
                { timeoutMs: TOKEN_REQUEST_TIMEOUT_MS }
              ),
              nowSeconds()
            )
        }
      )
    } catch (error) {
      if (error instanceof NativeLoginCancelledError) {
        log('[cloud] Hermes Cloud sign-in was cancelled')

        // A cancel changes nothing: an existing session stays signed in.
        return { signedIn: hasLivePortalSession(), cancelled: true }
      }

      throw error
    } finally {
      if (pending === flow) {
        pending = null
      }
    }

    // A newer sign-in (or a sign-out) started while this browser flow was
    // open: never let the older one overwrite it.
    if (!isCurrent()) {
      throw new NativeAuthChangedError()
    }

    coordinator.storeTokens(portalBaseUrl, tokens)
    log('[cloud] signed in to Hermes Cloud')
    deps.onSignedIn?.(portalTokenOrgId(tokens.accessToken))

    return { signedIn: true }
  }

  function logout(): void {
    coordinator.clearTokens(portal())
  }

  async function postExchange(subjectToken: string, agentId: string): Promise<NativeTokenSet> {
    return parseAgentTokenResponse(
      await deps.postJson(portalTokenUrl(portal()), agentTokenExchangeGrant({ subjectToken, agentId }), {
        timeoutMs: TOKEN_REQUEST_TIMEOUT_MS
      }),
      agentId,
      nowSeconds()
    )
  }

  // §5 is rate limited per desktop session: after a 429 no exchange (for any
  // agent) is sent before this time (monotonic seconds).
  let exchangeNotBefore = 0

  /**
   * §5: mint a bearer for one agent. `invalid_grant` is ambiguous (a stale
   * subject token OR a failed access gate), so it earns exactly ONE forced
   * portal refresh + retry; if it survives a fresh subject token the user
   * lost access. `invalid_target` is access lost outright. A 429 honours
   * Retry-After session-wide and is transient — never an auth verdict.
   * Never loops.
   */
  async function exchangeForAgent(agentId: string): Promise<NativeTokenSet> {
    const backoff = Math.ceil(exchangeNotBefore - monotonicSeconds())

    if (backoff > 0) {
      throw cloudExchangeRateLimitedError(backoff)
    }

    let subject = await getPortalAccessToken()

    if (!subject) {
      throw cloudLoginRequiredError(CLOUD_NOT_SIGNED_IN_MESSAGE)
    }

    for (let attempt = 0; ; attempt++) {
      try {
        return await postExchange(subject, agentId)
      } catch (error) {
        const code = oauthErrorCode(error)
        const status = readStatusCode(error)

        if (isPortalRateLimited(error)) {
          const wait = retryAfterSeconds(error, nowSeconds() * 1_000)

          exchangeNotBefore = monotonicSeconds() + wait
          log(`[cloud] Hermes Cloud token exchange rate limited; backing off ${wait}s`)

          throw cloudExchangeRateLimitedError(wait, error)
        }

        if (code === 'invalid_target') {
          throw cloudAgentAccessLostError(error)
        }

        const subjectMaybeStale = code === 'invalid_grant' || status === 401

        if (subjectMaybeStale && attempt === 0) {
          const rotated = await getPortalAccessToken({ forceRefresh: true, rejectedAccessToken: subject })

          if (!rotated) {
            throw cloudLoginRequiredError(CLOUD_SESSION_EXPIRED_MESSAGE, error)
          }

          if (rotated !== subject) {
            subject = rotated

            continue
          }
        }

        if (subjectMaybeStale || code === 'unauthorized_client') {
          throw cloudAgentAccessLostError(error)
        }

        throw error
      }
    }
  }

  return {
    hasLivePortalSession,
    getPortalAccessToken,
    login,
    cancelLogin,
    pendingAuthorizeUrl,
    logout,
    exchangeForAgent
  }
}

export type PortalSession = ReturnType<typeof createPortalSession>
