/**
 * native-oauth.ts
 *
 * Pure, electron-free helpers for the desktop's RFC 8252 (OAuth 2.0 for Native
 * Apps) login to a gated Hermes gateway: system-browser + loopback redirect +
 * PKCE, with tokens returned to the app (never browser session cookies).
 *
 * Kept standalone (no `import 'electron'`) so it unit-tests with `node --test`
 * — same pattern as connection-config.ts. main.ts owns the electron-coupled
 * parts (the actual http.Server loopback listener, shell.openExternal, and
 * safeStorage keychain writes) and calls these helpers for the pure logic.
 *
 * Why the gateway brokers the flow (not a direct desktop→IDP client): the
 * upstream IDP (Nous Portal) issues a per-gateway-instance client_id and only
 * accepts a redirect_uri on the gateway's own origin, so a desktop loopback
 * redirect can't be a direct Portal client. Instead the gateway exposes
 * /auth/native/{authorize,token,refresh}: it is the authorization server to
 * the desktop and an OAuth client to Portal. The desktop still gets the full
 * RFC 8252 experience — its own PKCE pair, its own loopback redirect, tokens
 * it stores itself.
 *
 * Capability detection: the gateway advertises supported flows on the public
 * /api/status `auth_flows` array. `native_pkce` present ⇒ use this flow;
 * absent (older gateway) ⇒ the caller falls back to the embedded-webview
 * cookie flow. This is the "observable ladder / compatibility fallback tied to
 * an identified older runtime" the desktop guide requires.
 */

import { createHash, randomBytes } from 'node:crypto'

// The gateway status field that lists supported auth flows. See
// hermes_cli/web_server.py status handler.
const NATIVE_FLOW_ID = 'native_pkce'

export interface NativePkcePair {
  verifier: string
  challenge: string
  method: 'S256'
}

export interface NativeTokenSet {
  accessToken: string
  refreshToken: string
  expiresAt: number
  provider: string
  userId: string
}

/** base64url without `=` padding (RFC 7636 §4). */
function b64url(raw: Buffer): string {
  return raw.toString('base64').replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')
}

/**
 * Generate a PKCE verifier/challenge pair (S256). The verifier is 32 random
 * bytes base64url-encoded (43 chars, within RFC 7636's 43–128 range).
 */
export function generatePkcePair(randomImpl: (n: number) => Buffer = randomBytes): NativePkcePair {
  const verifier = b64url(randomImpl(32))
  const challenge = b64url(createHash('sha256').update(verifier, 'ascii').digest())

  return { verifier, challenge, method: 'S256' }
}

/** A high-entropy CSRF `state` value for the loopback round trip. */
export function generateState(randomImpl: (n: number) => Buffer = randomBytes): string {
  return b64url(randomImpl(24))
}

/**
 * True if a gateway `/api/status` body advertises the native PKCE flow.
 * Tolerant of the field being absent (older gateway) or malformed.
 */
export function statusSupportsNativeFlow(statusBody: any): boolean {
  const flows = statusBody && statusBody.auth_flows

  return Array.isArray(flows) && flows.includes(NATIVE_FLOW_ID)
}

/**
 * Decide the login strategy for a gated gateway from its status body.
 * Returns 'native' when the gateway can do RFC 8252 AND we're not forced to
 * the legacy path; 'embedded' otherwise (older gateway ⇒ webview fallback).
 *
 * `forceEmbedded` lets a user/setting or an env override pin the legacy flow
 * (e.g. a corporate proxy that blocks loopback). Precedence written down here,
 * in one place, as a pure function — per the desktop "observable ladder" rule.
 */
export function resolveLoginStrategy(statusBody: any, opts: { forceEmbedded?: boolean } = {}): 'native' | 'embedded' {
  if (opts.forceEmbedded) {
    return 'embedded'
  }

  return statusSupportsNativeFlow(statusBody) ? 'native' : 'embedded'
}

/**
 * Build the gateway `/auth/native/authorize` URL the system browser opens.
 * `redirectUri` is the desktop's loopback callback (127.0.0.1:<port>/...).
 * `provider` is optional — omitted lets the gateway pick when it has exactly
 * one session provider (the common hosted case).
 */
export function buildNativeAuthorizeUrl(
  baseUrl: string,
  params: { challenge: string; redirectUri: string; state: string; provider?: string }
): string {
  const parsed = new URL(baseUrl)
  const prefix = parsed.pathname.replace(/\/+$/, '')

  const q = new URLSearchParams({
    code_challenge: params.challenge,
    code_challenge_method: 'S256',
    redirect_uri: params.redirectUri,
    state: params.state
  })

  if (params.provider) {
    q.set('provider', params.provider)
  }

  return `${parsed.protocol}//${parsed.host}${prefix}/auth/native/authorize?${q.toString()}`
}

/** The `/auth/native/token` endpoint URL for a gateway base URL. */
export function nativeTokenUrl(baseUrl: string): string {
  const parsed = new URL(baseUrl)
  const prefix = parsed.pathname.replace(/\/+$/, '')

  return `${parsed.protocol}//${parsed.host}${prefix}/auth/native/token`
}

/** The `/auth/native/refresh` endpoint URL for a gateway base URL. */
export function nativeRefreshUrl(baseUrl: string): string {
  const parsed = new URL(baseUrl)
  const prefix = parsed.pathname.replace(/\/+$/, '')

  return `${parsed.protocol}//${parsed.host}${prefix}/auth/native/refresh`
}

/**
 * Parse the loopback redirect the gateway sends the browser to. Returns the
 * `code` + `state`, or throws with the gateway's `error` if the flow failed.
 * `expectedState` MUST match (CSRF defense — RFC 6749 §10.12); a mismatch
 * throws rather than proceeding.
 */
export function parseLoopbackCallback(requestUrl: string, expectedState: string): { code: string } {
  // requestUrl is the path+query the loopback server received, e.g.
  // "/callback?code=...&state=...". Resolve against a dummy origin to parse.
  const parsed = new URL(requestUrl, 'http://127.0.0.1')
  const error = parsed.searchParams.get('error')

  if (error) {
    const desc = parsed.searchParams.get('error_description') || ''
    throw new Error(`Gateway rejected native login: ${error}${desc ? ` (${desc})` : ''}`)
  }

  const code = parsed.searchParams.get('code') || ''
  const state = parsed.searchParams.get('state') || ''

  if (!code) {
    throw new Error('Loopback callback missing authorization code')
  }

  if (!expectedState || state !== expectedState) {
    // Never redeem a code that arrived with a mismatched state — it may be a
    // forged callback trying to inject an attacker's code.
    throw new Error('Loopback callback state mismatch (possible CSRF)')
  }

  return { code }
}

/**
 * Normalize a `/auth/native/token` (or refresh) JSON response into a
 * NativeTokenSet, validating the shape. Throws on a missing/short access
 * token so a malformed response fails loudly rather than storing junk.
 */
export function parseTokenResponse(body: any): NativeTokenSet {
  const accessToken = String(body?.access_token || '')

  if (!accessToken) {
    throw new Error('Gateway token response missing access_token')
  }

  const expiresAt = Number(body?.expires_at)

  return {
    accessToken,
    refreshToken: String(body?.refresh_token || ''),
    expiresAt: Number.isFinite(expiresAt) ? expiresAt : 0,
    provider: String(body?.provider || ''),
    userId: String(body?.user_id || '')
  }
}

/**
 * True when a stored token set is at/near expiry and should be refreshed
 * before use. `skewSeconds` refreshes slightly early to avoid a race where
 * the token expires in flight (mirrors the server's 60s cookie floor).
 */
export function tokenNeedsRefresh(
  tokens: Pick<NativeTokenSet, 'expiresAt'>,
  nowSeconds: number,
  skewSeconds = 60
): boolean {
  if (!tokens || !Number.isFinite(tokens.expiresAt) || tokens.expiresAt <= 0) {
    // Unknown expiry ⇒ treat as needing refresh so we validate before use.
    return true
  }

  return nowSeconds >= tokens.expiresAt - skewSeconds
}

export { NATIVE_FLOW_ID }
