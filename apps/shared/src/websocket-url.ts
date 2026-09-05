export type GatewayAuthMode = 'oauth' | 'token' | (string & {})

export interface GatewayWsConnection {
  authMode?: GatewayAuthMode | null
  profile?: null | string
  wsUrl: string
}

export interface ResolveGatewayWsUrlDeps {
  /**
   * Returns a fresh WebSocket URL for the selected backend/profile.
   * OAuth-gated gateways use single-use tickets, so callers should mint
   * immediately before opening the socket.
   */
  getGatewayWsUrl?: (profile?: null | string) => Promise<GatewayWsUrlResult>
}

export type GatewayWsUrlResult =
  | string
  | { ok: true; wsUrl: string }
  | { error: string; needsOauthLogin?: boolean; ok: false }

export const GATEWAY_SESSION_TOKEN_REJECTED_MESSAGE =
  "This connection's session token was rejected. Open Settings -> Gateway and paste a new session token."

export class GatewayReauthRequiredError extends Error {
  readonly needsOauthLogin = true

  constructor(message: string, options?: { cause?: unknown }) {
    super(message, options)
    this.name = 'GatewayReauthRequiredError'
  }
}

export function isGatewayReauthRequired(error: unknown): error is GatewayReauthRequiredError {
  return (
    error instanceof GatewayReauthRequiredError ||
    (typeof error === 'object' && error !== null && (error as { needsOauthLogin?: unknown }).needsOauthLogin === true)
  )
}

/** Token/ticket handshake refused (HTTP 401/403 or gateway close 4401). */
const GATEWAY_AUTH_FAILURE_RE =
  /\b(401|403|4401)\b|unauthorized|session token was rejected|needs oauth login/i

export function isGatewayHandshakeAuthFailure(error: unknown): boolean {
  if (isGatewayReauthRequired(error)) {
    return true
  }

  if (error && typeof error === 'object') {
    const record = error as { error?: unknown; needsSessionToken?: unknown; statusCode?: unknown }
    if (record.needsSessionToken === true) {
      return true
    }

    const status = Number(record.statusCode)

    if (status === 401 || status === 403 || status === 4401) {
      return true
    }

    if (typeof record.error === 'string' && GATEWAY_AUTH_FAILURE_RE.test(record.error)) {
      return true
    }
  }

  const text = error instanceof Error ? error.message : String(error ?? '')

  return GATEWAY_AUTH_FAILURE_RE.test(text)
}

export async function resolveGatewayWsUrl(deps: ResolveGatewayWsUrlDeps, conn: GatewayWsConnection): Promise<string> {
  const mint = deps.getGatewayWsUrl
  const profile = conn.profile ?? null

  if (conn.authMode === 'oauth') {
    if (!mint) {
      throw new Error('This Desktop build cannot refresh OAuth WebSocket tickets. Update Hermes Desktop and try again.')
    }

    try {
      const result = await mint(profile)

      if (typeof result === 'string') {
        return result
      }

      if (result.ok) {
        return result.wsUrl
      }

      if (result.needsOauthLogin) {
        throw new GatewayReauthRequiredError(
          'Your remote gateway session has expired. Open Settings -> Gateway and click "Sign in" again.',
          { cause: new Error(result.error) }
        )
      }

      throw new Error(result.error || 'Could not refresh the remote gateway WebSocket ticket.')
    } catch (error) {
      if (isGatewayReauthRequired(error)) {
        throw error instanceof GatewayReauthRequiredError
          ? error
          : new GatewayReauthRequiredError(
              'Your remote gateway session has expired. Open Settings -> Gateway and click "Sign in" again.',
              { cause: error }
            )
      }

      throw error
    }
  }

  if (mint) {
    try {
      const fresh = await mint(profile)

      if (typeof fresh === 'string') {
        return fresh
      }

      if (fresh?.ok) {
        return fresh.wsUrl
      }

      if (fresh && isGatewayHandshakeAuthFailure(fresh)) {
        throw new GatewayReauthRequiredError(GATEWAY_SESSION_TOKEN_REJECTED_MESSAGE, {
          cause: new Error(fresh.error)
        })
      }
    } catch (error) {
      if (isGatewayReauthRequired(error)) {
        throw error
      }

      if (isGatewayHandshakeAuthFailure(error)) {
        throw new GatewayReauthRequiredError(GATEWAY_SESSION_TOKEN_REJECTED_MESSAGE, {
          cause: error instanceof Error ? error : undefined
        })
      }
    }
  }

  return conn.wsUrl
}

export type WebSocketAuthParam = readonly [name: string, value: string]

export interface HermesWebSocketUrlOptions {
  /** Dashboard or gateway-relative endpoint path, e.g. "/api/ws". */
  path: string
  /** Optional URL prefix when the backend is reverse-proxied below a subpath. */
  basePath?: string
  /** Query auth pair, usually ["token", value] or ["ticket", value]. */
  authParam?: WebSocketAuthParam
  /** Extra query params merged before auth. */
  params?: Record<string, string>
  /** Browser protocol string such as "https:"; defaults to window.location.protocol. */
  protocol?: string
  /** Host with optional port; defaults to window.location.host. */
  host?: string
}

function readWindowLocation(): { host: string; protocol: string } {
  if (typeof window === 'undefined') {
    return { host: '', protocol: 'http:' }
  }

  return { host: window.location.host, protocol: window.location.protocol }
}

function normalizeBasePath(basePath: string | undefined): string {
  if (!basePath) {
    return ''
  }

  const withLead = basePath.startsWith('/') ? basePath : `/${basePath}`

  return withLead.replace(/\/+$/, '')
}

function normalizeEndpointPath(path: string): string {
  return path.startsWith('/') ? path : `/${path}`
}

export function buildHermesWebSocketUrl(options: HermesWebSocketUrlOptions): string {
  const loc = readWindowLocation()
  const protocol = options.protocol ?? loc.protocol
  const host = options.host ?? loc.host
  const wsScheme = protocol === 'https:' || protocol === 'wss:' ? 'wss:' : 'ws:'
  const qs = new URLSearchParams(options.params ?? {})

  if (options.authParam) {
    const [name, value] = options.authParam
    qs.set(name, value)
  }

  const query = qs.toString()
  const suffix = query ? `?${query}` : ''

  return `${wsScheme}//${host}${normalizeBasePath(options.basePath)}${normalizeEndpointPath(options.path)}${suffix}`
}
