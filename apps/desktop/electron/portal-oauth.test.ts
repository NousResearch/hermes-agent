/**
 * Wire-shape tests for the Hermes Desktop ⇄ portal OAuth client
 * (`hermes-desktop`). Every literal here is pinned by the locked contract
 * (specs/desktop-native-cloud-login/contract.md §1, §2, §3, §5); a failure
 * means the desktop drifted from what the portal accepts.
 */

import { expect, test } from 'vitest'

import { httpStatusError } from './api-transport'
import {
  agentTokenExchangeGrant,
  authorizationCodeGrant,
  buildPortalAuthorizeUrl,
  CLOUD_AGENT_TOKEN_PROVIDER,
  isPortalRateLimited,
  oauthErrorCode,
  parseAgentTokenResponse,
  parsePortalTokenResponse,
  PORTAL_TOKEN_PROVIDER,
  portalTokenOrgId,
  portalTokenUrl,
  refreshTokenGrant,
  retryAfterSeconds
} from './portal-oauth'

const PORTAL = 'https://portal.example.test'

function jwt(claims: Record<string, unknown>) {
  const enc = (value: unknown) => Buffer.from(JSON.stringify(value)).toString('base64url')

  return `${enc({ alg: 'RS256', typ: 'JWT' })}.${enc(claims)}.sig`
}

test('§1 authorize URL carries exactly the contract params (client_id, S256, scope, loopback redirect)', () => {
  const url = buildPortalAuthorizeUrl(PORTAL, {
    challenge: 'CHALLENGE_b64url',
    redirectUri: 'http://127.0.0.1:53123/callback',
    state: 'STATE123'
  })

  const parsed = new URL(url)
  expect(`${parsed.origin}${parsed.pathname}`).toBe(`${PORTAL}/oauth/authorize`)
  expect(Object.fromEntries(parsed.searchParams)).toEqual({
    response_type: 'code',
    client_id: 'hermes-desktop',
    redirect_uri: 'http://127.0.0.1:53123/callback',
    state: 'STATE123',
    code_challenge: 'CHALLENGE_b64url',
    code_challenge_method: 'S256',
    scope: 'agents:read agents:connect'
  })
  // The contract spells the space as %20, never '+'.
  expect(url).toContain('scope=agents%3Aread%20agents%3Aconnect')
  expect(url).not.toContain('+')
  // No org param from the desktop: the portal's picker owns team choice.
  expect(parsed.searchParams.has('org')).toBe(false)
})

test('§1 authorize URL tolerates a trailing slash on the portal base', () => {
  const url = buildPortalAuthorizeUrl(`${PORTAL}/`, {
    challenge: 'c',
    redirectUri: 'http://127.0.0.1:1/callback',
    state: 's'
  })

  expect(url.startsWith(`${PORTAL}/oauth/authorize?`)).toBe(true)
})

test('token endpoint is {portal}/api/oauth/token', () => {
  expect(portalTokenUrl(PORTAL)).toBe(`${PORTAL}/api/oauth/token`)
  expect(portalTokenUrl(`${PORTAL}//`)).toBe(`${PORTAL}/api/oauth/token`)
})

test('§2 code → token body is exactly the contract fields, incl. redirect_uri', () => {
  expect(
    authorizationCodeGrant({ code: 'CODE', codeVerifier: 'VERIFIER', redirectUri: 'http://127.0.0.1:53123/callback' })
  ).toEqual({
    grant_type: 'authorization_code',
    client_id: 'hermes-desktop',
    code: 'CODE',
    code_verifier: 'VERIFIER',
    redirect_uri: 'http://127.0.0.1:53123/callback'
  })
})

test('§3 refresh body is exactly the contract fields', () => {
  expect(refreshTokenGrant('RT-1')).toEqual({
    grant_type: 'refresh_token',
    client_id: 'hermes-desktop',
    refresh_token: 'RT-1'
  })
})

test('§5 token-exchange body is exactly the contract literals', () => {
  expect(agentTokenExchangeGrant({ subjectToken: 'DESKTOP_AT', agentId: 'agt_123' })).toEqual({
    grant_type: 'urn:ietf:params:oauth:grant-type:token-exchange',
    client_id: 'hermes-desktop',
    subject_token: 'DESKTOP_AT',
    subject_token_type: 'urn:ietf:params:oauth:token-type:access_token',
    audience: 'agent:agt_123'
  })
})

test('§2/§3 token response parses expires_in into an absolute expiry and keeps the refresh token', () => {
  const tokens = parsePortalTokenResponse(
    {
      access_token: 'AT',
      token_type: 'Bearer',
      expires_in: 900,
      refresh_token: 'RT',
      scope: 'agents:read agents:connect'
    },
    1_000
  )

  expect(tokens).toEqual({
    accessToken: 'AT',
    refreshToken: 'RT',
    expiresAt: 1_900,
    provider: PORTAL_TOKEN_PROVIDER,
    userId: ''
  })
})

test('a portal token response without access_token or refresh_token fails loudly', () => {
  expect(() => parsePortalTokenResponse({ refresh_token: 'RT', expires_in: 900 }, 0)).toThrow(/access_token/)
  expect(() => parsePortalTokenResponse({ access_token: 'AT', expires_in: 900 }, 0)).toThrow(/refresh_token/)
})

test('a missing expires_in falls back to the JWT exp, never to "always expired"', () => {
  const at = jwt({ exp: 5_000 })

  expect(parsePortalTokenResponse({ access_token: at, refresh_token: 'RT' }, 1_000).expiresAt).toBe(5_000)
  expect(parsePortalTokenResponse({ access_token: 'opaque', refresh_token: 'RT' }, 1_000).expiresAt).toBeGreaterThan(
    1_000
  )
})

test('§5 agent token response is tagged as a cloud-agent token carrying its agent id and no refresh token', () => {
  const tokens = parseAgentTokenResponse(
    {
      access_token: 'AGENT_AT',
      issued_token_type: 'urn:ietf:params:oauth:token-type:access_token',
      token_type: 'Bearer',
      expires_in: 900,
      scope: 'agent_dashboard:access'
    },
    'agt_123',
    100
  )

  expect(tokens).toEqual({
    accessToken: 'AGENT_AT',
    refreshToken: '',
    expiresAt: 1_000,
    provider: CLOUD_AGENT_TOKEN_PROVIDER,
    userId: 'agt_123'
  })
  expect(() => parseAgentTokenResponse({ expires_in: 900 }, 'agt_123', 0)).toThrow(/access_token/)
})

test('oauthErrorCode reads the RFC 6749 error code off a thrown HTTP error', () => {
  expect(oauthErrorCode(httpStatusError(400, JSON.stringify({ error: 'invalid_grant', error_description: 'x' })))).toBe(
    'invalid_grant'
  )
  expect(oauthErrorCode(httpStatusError(502, '<html>bad gateway</html>'))).toBeNull()
  expect(oauthErrorCode(new Error('socket hang up'))).toBeNull()
})

test('portalTokenOrgId reads org_id from the access token for display only', () => {
  expect(portalTokenOrgId(jwt({ org_id: 'org_9', client_id: 'hermes-desktop' }))).toBe('org_9')
  expect(portalTokenOrgId('not-a-jwt')).toBeNull()
  expect(portalTokenOrgId(jwt({ sub: 'u' }))).toBeNull()
})

test('§5 rate limit: 429 or slow_down is rate limited; Retry-After reads delta-seconds or an HTTP date, clamped', () => {
  const limited = (retryAfter?: string) =>
    Object.assign(httpStatusError(429, JSON.stringify({ error: 'slow_down' })), retryAfter ? { retryAfter } : {})

  expect(isPortalRateLimited(limited())).toBe(true)
  expect(isPortalRateLimited(httpStatusError(400, JSON.stringify({ error: 'slow_down' })))).toBe(true)
  expect(isPortalRateLimited(httpStatusError(400, JSON.stringify({ error: 'invalid_grant' })))).toBe(false)

  const now = Date.parse('2026-09-24T00:00:00Z')
  expect(retryAfterSeconds(limited('17'), now)).toBe(17)
  expect(retryAfterSeconds(limited('Thu, 24 Sep 2026 00:00:45 GMT'), now)).toBe(45)
  expect(retryAfterSeconds(limited(), now)).toBe(60)
  expect(retryAfterSeconds(limited('soon'), now)).toBe(60)
  expect(retryAfterSeconds(limited('0'), now)).toBe(1)
  expect(retryAfterSeconds(limited('999999'), now)).toBe(900)
})
