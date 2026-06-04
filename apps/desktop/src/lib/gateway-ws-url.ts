import type { HermesConnection } from '@/global'

/**
 * The desktop main process exposes `getGatewayWsUrl()` to re-mint a WebSocket
 * URL immediately before every `gateway.connect()`. For OAuth-gated remote
 * gateways the WS ticket is single-use with a ~30s TTL, so the ticket baked
 * into the cached `conn.wsUrl` is stale (and, after the first connect, already
 * consumed). For local/token gateways the URL carries a long-lived token and
 * never needs re-minting.
 *
 * Resolution rules:
 *
 * - OAuth: the fresh mint is the *only* viable URL. If it fails, do NOT fall
 *   back to `conn.wsUrl` — that ticket is dead and the connect is guaranteed to
 *   fail with an opaque "connection closed" error. Instead, let the mint error
 *   propagate so the caller can surface the gateway's reauth message
 *   ("session has expired… Sign in again").
 *
 * - token / local, or when the preload method is genuinely absent (older
 *   preload shapes): fall back to `conn.wsUrl`. The token URL is long-lived, so
 *   the fallback is safe and preserves compatibility.
 *
 * The error thrown for OAuth mint failures is tagged with `needsOauthLogin` so
 * callers can distinguish "the user must re-authenticate" from a generic
 * transport failure.
 */
export interface ResolveGatewayWsUrlDeps {
  /** `window.hermesDesktop.getGatewayWsUrl`, if the preload exposes it. */
  getGatewayWsUrl?: () => Promise<string>
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

export async function resolveGatewayWsUrl(
  desktop: ResolveGatewayWsUrlDeps,
  conn: Pick<HermesConnection, 'authMode' | 'wsUrl'>
): Promise<string> {
  const mint = desktop.getGatewayWsUrl

  if (conn.authMode === 'oauth') {
    if (!mint) {
      // OAuth gateway but no way to mint a fresh ticket: the cached ticket is
      // dead, so connecting with it cannot succeed. Surface a reauth error
      // rather than silently attempting a doomed connect.
      throw new GatewayReauthRequiredError(
        'Your remote gateway session needs to be refreshed. Open Settings → Gateway and click "Sign in" again.'
      )
    }

    try {
      return await mint()
    } catch (error) {
      throw new GatewayReauthRequiredError(
        'Your remote gateway session has expired. Open Settings → Gateway and click "Sign in" again.',
        { cause: error }
      )
    }
  }

  // token / local: the URL carries a long-lived token. Re-mint when available
  // (cheap, keeps parity), but the cached URL is a safe fallback.
  if (mint) {
    const fresh = await mint().catch(() => null)

    if (fresh) {
      return fresh
    }
  }

  return conn.wsUrl
}
