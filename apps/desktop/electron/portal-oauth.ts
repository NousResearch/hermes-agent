/**
 * portal-oauth.ts
 *
 * Pure, electron-free wire helpers for Hermes Desktop as the portal's public
 * OAuth client `hermes-desktop` (RFC 8252 loopback + PKCE S256). Every literal
 * here is fixed by the desktop ⇄ portal wire contract:
 *
 *   §1 GET  {portal}/oauth/authorize          (system browser)
 *   §2 POST {portal}/api/oauth/token          authorization_code → AT + RT
 *   §3 POST {portal}/api/oauth/token          refresh_token (rotating)
 *   §5 POST {portal}/api/oauth/token          RFC 8693 token exchange →
 *                                             agent dashboard bearer
 *
 * The portal accepts form-encoded OR JSON token requests; the desktop sends
 * JSON through the same cookieless transport the gateway native flow uses.
 */

import { readJsonErrorBody, readStatusCode } from './api-transport'
import type { NativeTokenSet } from './native-oauth'

export const PORTAL_CLIENT_ID = 'hermes-desktop'
export const PORTAL_SCOPE = 'agents:read agents:connect'
export const TOKEN_EXCHANGE_GRANT_TYPE = 'urn:ietf:params:oauth:grant-type:token-exchange'
export const ACCESS_TOKEN_TYPE = 'urn:ietf:params:oauth:token-type:access_token'

/** NativeTokenSet.provider marker for the stored portal (desktop) session. */
export const PORTAL_TOKEN_PROVIDER = 'hermes-desktop'
/**
 * NativeTokenSet.provider marker for an agent bearer minted by §5. Such a set
 * has no refresh token; its `userId` slot carries the AgentInstance id so a
 * re-exchange knows its audience after a restart.
 */
export const CLOUD_AGENT_TOKEN_PROVIDER = 'hermes-cloud-agent'

// When a token response omits expires_in and the token is not a readable JWT,
// assume a short life rather than "already expired" (which would refresh on
// every request) or "never" (which would ride a dead token).
const FALLBACK_TTL_SECONDS = 300

function trimBase(portalBaseUrl: string): string {
  return String(portalBaseUrl).trim().replace(/\/+$/, '')
}

/** §1 — the URL the system browser opens. Space in scope is %20, per contract. */
export function buildPortalAuthorizeUrl(
  portalBaseUrl: string,
  params: { challenge: string; redirectUri: string; state: string }
): string {
  const query = [
    ['response_type', 'code'],
    ['client_id', PORTAL_CLIENT_ID],
    ['redirect_uri', params.redirectUri],
    ['state', params.state],
    ['code_challenge', params.challenge],
    ['code_challenge_method', 'S256'],
    ['scope', PORTAL_SCOPE]
  ]
    .map(([key, value]) => `${key}=${encodeURIComponent(value)}`)
    .join('&')

  return `${trimBase(portalBaseUrl)}/oauth/authorize?${query}`
}

/** §2/§3/§5 — the single token endpoint. */
export function portalTokenUrl(portalBaseUrl: string): string {
  return `${trimBase(portalBaseUrl)}/api/oauth/token`
}

/** §2 — authorization code → tokens. redirect_uri must equal the authorize one. */
export function authorizationCodeGrant(params: { code: string; codeVerifier: string; redirectUri: string }) {
  return {
    grant_type: 'authorization_code',
    client_id: PORTAL_CLIENT_ID,
    code: params.code,
    code_verifier: params.codeVerifier,
    redirect_uri: params.redirectUri
  }
}

/** §3 — rotate the desktop session. */
export function refreshTokenGrant(refreshToken: string) {
  return {
    grant_type: 'refresh_token',
    client_id: PORTAL_CLIENT_ID,
    refresh_token: refreshToken
  }
}

/** §5 audience for a Hermes Cloud AgentInstance. */
export function agentAudience(agentId: string): string {
  return `agent:${agentId}`
}

/** §5 — desktop access token → agent dashboard token (RFC 8693). */
export function agentTokenExchangeGrant(params: { subjectToken: string; agentId: string }) {
  return {
    grant_type: TOKEN_EXCHANGE_GRANT_TYPE,
    client_id: PORTAL_CLIENT_ID,
    subject_token: params.subjectToken,
    subject_token_type: ACCESS_TOKEN_TYPE,
    audience: agentAudience(params.agentId)
  }
}

/** Unverified JWT claims — for expiry fallbacks and display only, never trust. */
function jwtClaims(token: string): null | Record<string, unknown> {
  const parts = String(token || '').split('.')

  if (parts.length !== 3) {
    return null
  }

  try {
    const parsed: unknown = JSON.parse(Buffer.from(parts[1], 'base64url').toString('utf8'))

    return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? (parsed as Record<string, unknown>) : null
  } catch {
    return null
  }
}

function expiresAtFrom(body: any, accessToken: string, nowSeconds: number): number {
  const expiresIn = Number(body?.expires_in)

  if (Number.isFinite(expiresIn) && expiresIn > 0) {
    return nowSeconds + Math.floor(expiresIn)
  }

  const exp = Number(jwtClaims(accessToken)?.exp)

  return Number.isFinite(exp) && exp > 0 ? exp : nowSeconds + FALLBACK_TTL_SECONDS
}

/** §2/§3 200 body → the stored desktop session. Both tokens are mandatory. */
export function parsePortalTokenResponse(body: any, nowSeconds: number): NativeTokenSet {
  const accessToken = String(body?.access_token || '')
  const refreshToken = String(body?.refresh_token || '')

  if (!accessToken) {
    throw new Error('Hermes Cloud token response missing access_token')
  }

  if (!refreshToken) {
    throw new Error('Hermes Cloud token response missing refresh_token')
  }

  return {
    accessToken,
    refreshToken,
    expiresAt: expiresAtFrom(body, accessToken, nowSeconds),
    provider: PORTAL_TOKEN_PROVIDER,
    userId: ''
  }
}

/** §5 200 body → an agent bearer tagged with its agent id (no refresh token). */
export function parseAgentTokenResponse(body: any, agentId: string, nowSeconds: number): NativeTokenSet {
  const accessToken = String(body?.access_token || '')

  if (!accessToken) {
    throw new Error('Hermes Cloud token exchange response missing access_token')
  }

  return {
    accessToken,
    refreshToken: '',
    expiresAt: expiresAtFrom(body, accessToken, nowSeconds),
    provider: CLOUD_AGENT_TOKEN_PROVIDER,
    userId: agentId
  }
}

/** The RFC 6749 `error` code a thrown portal HTTP error carries, if any. */
export function oauthErrorCode(error: unknown): null | string {
  const code = readJsonErrorBody(error)?.error

  return typeof code === 'string' && code ? code : null
}

// §5 429 without a usable Retry-After: back off a minute (the limit is 60/min).
const DEFAULT_RETRY_AFTER_SECONDS = 60
const MAX_RETRY_AFTER_SECONDS = 15 * 60

/** A §5 rate-limit answer: 429, or the RFC 8628-style `slow_down` code. */
export function isPortalRateLimited(error: unknown): boolean {
  return readStatusCode(error) === 429 || oauthErrorCode(error) === 'slow_down'
}

/**
 * Seconds to wait from the `Retry-After` header an HTTP error carries (the
 * transport copies it onto `error.retryAfter`): delta-seconds or an HTTP
 * date. Missing/unparseable → a conservative default; clamped to [1, 15 min].
 */
export function retryAfterSeconds(error: unknown, nowMs: number): number {
  const raw =
    error && typeof error === 'object' ? String((error as { retryAfter?: unknown }).retryAfter ?? '').trim() : ''

  let seconds = DEFAULT_RETRY_AFTER_SECONDS

  if (/^\d+$/.test(raw)) {
    seconds = Number(raw)
  } else if (raw) {
    const at = Date.parse(raw)

    if (Number.isFinite(at)) {
      seconds = Math.ceil((at - nowMs) / 1_000)
    }
  }

  return Math.min(MAX_RETRY_AFTER_SECONDS, Math.max(1, seconds))
}

/** The org the desktop token is pinned to (`org_id` claim), for display only. */
export function portalTokenOrgId(accessToken: string): null | string {
  const orgId = jwtClaims(accessToken)?.org_id

  return typeof orgId === 'string' && orgId ? orgId : null
}
