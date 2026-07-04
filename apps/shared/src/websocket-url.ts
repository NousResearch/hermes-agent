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
  getGatewayWsUrl?: (profile?: null | string) => Promise<string>
}

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

// HTTP auth statuses embedded in gateway/main-process error messages, e.g.
// "401: {...}" from fetchJson or "status 403" from the HTML guard. Errors that
// cross Electron's IPC bridge keep only their message (statusCode and
// needsOauthLogin are stripped), so message inspection is the only renderer-side
// signal. Word-boundary anchored so ports/ids containing "401" don't match.
const AUTH_STATUS_PATTERN = /(?:^|[^0-9])(?:401|403)(?:[^0-9]|$)/
// Canonical reauth phrasings the main process throws only for genuine
// sign-in-required failures (buildRemoteConnection / mintGatewayWsTicket).
const AUTH_MESSAGE_PATTERN = /session has expired|not signed in|sign in/i

/** True when an error (possibly IPC-flattened to just a message) indicates the
 *  gateway rejected our credentials — as opposed to a transport failure
 *  (timeout, unreachable host) that a retry can recover from. */
export function isGatewayAuthShapedError(error: unknown): boolean {
  if (isGatewayReauthRequired(error)) {
    return true
  }

  if (typeof error === 'object' && error !== null) {
    const statusCode = (error as { statusCode?: unknown }).statusCode

    if (statusCode === 401 || statusCode === 403) {
      return true
    }
  }

  const message = error instanceof Error ? error.message : typeof error === 'string' ? error : ''

  return AUTH_STATUS_PATTERN.test(message) || AUTH_MESSAGE_PATTERN.test(message)
}

export async function resolveGatewayWsUrl(deps: ResolveGatewayWsUrlDeps, conn: GatewayWsConnection): Promise<string> {
  const mint = deps.getGatewayWsUrl
  const profile = conn.profile ?? null

  if (conn.authMode === 'oauth') {
    if (!mint) {
      throw new GatewayReauthRequiredError(
        'Your remote gateway session needs to be refreshed. Open Settings -> Gateway and click "Sign in" again.'
      )
    }

    try {
      return await mint(profile)
    } catch (error) {
      // Only a credential rejection means the OAuth session is dead. A
      // transport failure (Tailscale blip, sleep/wake race, backend restart)
      // must stay a plain retryable error — wrapping it as reauth here sent
      // users to a needless re-sign-in while the backoff loop would have
      // recovered on its own.
      if (isGatewayAuthShapedError(error)) {
        throw new GatewayReauthRequiredError(
          'Your remote gateway session has expired. Open Settings -> Gateway and click "Sign in" again.',
          { cause: error }
        )
      }

      throw error
    }
  }

  if (mint) {
    const fresh = await mint(profile).catch(() => null)

    if (fresh) {
      return fresh
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
