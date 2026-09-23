import { waitForHermesReady } from './backend-health'
import { resolveReadinessProbeAuth } from './native-auth-decisions'
import { requestWithOauthFallback } from './oauth-rest-request'

interface DesktopGatewayReadinessDeps {
  fetchJson: any
  fetchPublicJson: any
  fetchJsonViaOauthSession: any
  ensureNativeAccessToken: any
}

export function createDesktopGatewayReadinessRuntime(deps: DesktopGatewayReadinessDeps) {
  const { fetchJson, fetchPublicJson, fetchJsonViaOauthSession, ensureNativeAccessToken } = deps

  function requestOptionsWithHeaders(options: any = {}, headers = {}) {
    return {
      ...options,
      headers: {
        ...headers,
        ...(options.headers || {})
      }
    }
  }

  // Best-effort read of a gateway's advertised auth providers, cached per base
  // URL for the life of the process. Used by the oauth pre-flight guard to tell
  // a password-provider gateway (which cannot satisfy the bearer/cookie checks
  // by design) from a real OAuth one. Any failure returns [] so callers keep the
  // strict guard — backends predating /api/auth/providers are unaffected.
  const gatewayAuthProvidersCache = new Map<string, any[]>()

  async function gatewayAuthProviders(baseUrl, headers = {}) {
    const cached = gatewayAuthProvidersCache.get(baseUrl)

    if (cached) {
      return cached
    }

    let providers = []

    try {
      const body = (await fetchPublicJson(
        `${baseUrl}/api/auth/providers`,
        requestOptionsWithHeaders({ timeoutMs: 8_000 }, headers)
      )) as any

      if (Array.isArray(body?.providers)) {
        providers = body.providers
          .filter(p => p && typeof p === 'object')
          .map(p => ({ name: String(p.name || ''), supportsPassword: Boolean(p.supports_password) }))
          .filter(p => p.name)
      }

      gatewayAuthProvidersCache.set(baseUrl, providers)
    } catch {
      // Optional metadata — an unreadable list keeps the strict guard.
    }

    return providers
  }

  // Build the readiness probe for a connection's auth mode. A gated gateway
  // must be probed with the SAME credentials the rest of the connection uses:
  // an anonymous probe 401s forever against a live session, and it can never
  // see the 404 that identifies a backend predating /api/health (the auth gate
  // answers before the SPA catch-all). `probeIsCredentialed` tells
  // waitForHermesReady how to read a 401 — rejected session vs gated route.
  async function buildReadinessHealthProbe(baseUrl, authMode, token) {
    if (authMode === 'oauth') {
      return {
        probeHealth: (url: string, options: any = {}) =>
          requestWithOauthFallback(baseUrl, {
            ensureNativeAccessToken,
            requestWithBearer: bearer => fetchJson(url, null, { ...options, bearer }),
            requestWithCookie: () => fetchJsonViaOauthSession(url, options)
          }),
        probeIsCredentialed: true
      }
    }

    const probeAuth = resolveReadinessProbeAuth(authMode, null, token)

    if (probeAuth.kind === 'token' && probeAuth.token) {
      return {
        probeHealth: (url, options: any = {}) => fetchJson(url, probeAuth.token, options),
        probeIsCredentialed: true
      }
    }

    return { probeHealth: fetchPublicJson, probeIsCredentialed: false }
  }

  async function waitForHermes(baseUrl, token, signal?, authMode?, headers = {}) {
    const { probeHealth, probeIsCredentialed } = await buildReadinessHealthProbe(baseUrl, authMode, token)

    return waitForHermesReady(baseUrl, {
      token,
      signal,
      fetchPublicJson,
      fetchJson: probeIsCredentialed
        ? (url, _token, options = {}) => probeHealth(url, requestOptionsWithHeaders(options, headers))
        : fetchJson,
      probeHealth: (url, options = {}) => probeHealth(url, requestOptionsWithHeaders(options, headers)),
      probeIsCredentialed
    })
  }

  return { gatewayAuthProviders, waitForHermes }
}
