/**
 * auth.ts — connect/probe/login against a remote gateway.
 *
 * Sequence (verified against hermes_cli/dashboard_auth/routes.py):
 *   1. probeGateway()  → GET /api/status (+ /api/auth/providers) → auth mode
 *   2. passwordLogin() → POST /auth/password-login {provider,username,password}
 *                        → 200 {ok,next} + HttpOnly session cookies
 *   3. mintWsTicket()  → POST /api/auth/ws-ticket → {ticket, ttl_seconds}
 */

import { authModeFromStatus, normalizeRemoteBaseUrl, type AuthMode } from './connection-config'
import { clearCookies, hasSession, loadCookies } from './cookie-jar'
import { rawRequest } from './http'

export interface AuthProvider {
  name: string
  displayName: string
  supportsPassword: boolean
}

export interface ProbeResult {
  baseUrl: string
  reachable: boolean
  authMode: AuthMode
  /** True when a gateway is reachable but requires login (gated). */
  needsLogin: boolean
  providers: AuthProvider[]
  version: string | null
  error: string | null
}

/** Normalize + probe a user-entered URL. Never throws on network failure —
 *  returns `reachable:false` so the connect screen can show a friendly error. */
export async function probeGateway(rawUrl: string): Promise<ProbeResult> {
  let baseUrl: string
  try {
    baseUrl = normalizeRemoteBaseUrl(rawUrl)
  } catch (e) {
    return blankProbe(rawUrl, (e as Error).message)
  }

  await loadCookies(baseUrl)

  let status: Record<string, unknown>
  try {
    const res = await rawRequest({ baseUrl, path: '/api/status', timeoutMs: 10_000 })
    if (res.status < 200 || res.status >= 300) {
      return { ...blankProbe(baseUrl, `Gateway returned HTTP ${res.status}`), reachable: false }
    }
    status = (res.json ?? {}) as Record<string, unknown>
  } catch (e) {
    return blankProbe(baseUrl, `Could not reach gateway: ${(e as Error).message}`)
  }

  const authMode = authModeFromStatus(status)
  const providers = authMode === 'oauth' ? await fetchProviders(baseUrl) : []
  const needsLogin = authMode === 'oauth' && !hasSession(baseUrl)

  return {
    baseUrl,
    reachable: true,
    authMode,
    needsLogin,
    providers,
    version: typeof status.version === 'string' ? status.version : null,
    error: null,
  }
}

async function fetchProviders(baseUrl: string): Promise<AuthProvider[]> {
  try {
    const res = await rawRequest({ baseUrl, path: '/api/auth/providers', timeoutMs: 10_000 })
    if (res.status < 200 || res.status >= 300) return []
    const body = (res.json ?? {}) as {
      providers?: Array<{ name: string; display_name?: string; supports_password?: boolean }>
    }
    return (body.providers ?? []).map((p) => ({
      name: p.name,
      displayName: p.display_name ?? p.name,
      supportsPassword: Boolean(p.supports_password),
    }))
  } catch {
    return []
  }
}

export class LoginError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message)
    this.name = 'LoginError'
  }
}

/** Username/password → session cookies (captured into the jar by rawRequest). */
export async function passwordLogin(
  baseUrl: string,
  opts: { provider: string; username: string; password: string },
): Promise<void> {
  const res = await rawRequest({
    baseUrl,
    path: '/auth/password-login',
    method: 'POST',
    body: { provider: opts.provider, username: opts.username, password: opts.password, next: '/' },
    timeoutMs: 15_000,
  })

  if (res.status >= 200 && res.status < 300) {
    if (!hasSession(baseUrl)) {
      // 200 but no cookie captured — almost always means CapacitorHttp didn't
      // surface Set-Cookie (R2). Surface a clear, actionable error.
      throw new LoginError(
        'Logged in but no session cookie was returned. (Set-Cookie not captured.)',
        res.status,
      )
    }
    return
  }

  const detail =
    (res.json as { detail?: string } | null)?.detail ?? res.text ?? 'Login failed'
  if (res.status === 401) throw new LoginError('Invalid username or password.', 401)
  if (res.status === 404) throw new LoginError('This gateway has no password login.', 404)
  if (res.status === 429) throw new LoginError('Too many attempts — wait a moment.', 429)
  throw new LoginError(`${res.status}: ${detail}`, res.status)
}

/** Mint a single-use WS ticket (30s TTL). Must be called fresh per connect. */
export async function mintWsTicket(baseUrl: string): Promise<string> {
  const res = await rawRequest({
    baseUrl,
    path: '/api/auth/ws-ticket',
    method: 'POST',
    timeoutMs: 10_000,
  })
  if (res.status < 200 || res.status >= 300) {
    throw new Error(`ws-ticket: HTTP ${res.status}`)
  }
  const ticket = (res.json as { ticket?: string } | null)?.ticket
  if (!ticket) throw new Error('ws-ticket: gateway returned no ticket')
  return ticket
}

/** Best-effort logout: clear local cookies and tell the gateway to revoke. */
export async function logout(baseUrl: string): Promise<void> {
  try {
    await rawRequest({ baseUrl, path: '/auth/logout', method: 'POST', timeoutMs: 8_000 })
  } catch {
    /* best effort */
  }
  await clearCookies(baseUrl)
}

function blankProbe(baseUrl: string, error: string): ProbeResult {
  return {
    baseUrl,
    reachable: false,
    authMode: 'token',
    needsLogin: false,
    providers: [],
    version: null,
    error,
  }
}
