const CF_ACCESS_CLIENT_ID_HEADER = 'CF-Access-Client-Id'
const CF_ACCESS_CLIENT_SECRET_HEADER = 'CF-Access-Client-Secret'

type RequestHeaders = Record<string, string>

/**
 * Build the standard Cloudflare Access service-token header pair.
 *
 * Both values are required together. Returning an empty object for two empty
 * values keeps the feature opt-in; accepting only half of the pair would make
 * every request fail at the proxy with a misleading gateway error.
 */
function cloudflareAccessHeaders(clientId: unknown, clientSecret: unknown): RequestHeaders {
  const id = String(clientId || '').trim()
  const secret = String(clientSecret || '').trim()

  if (!id && !secret) {
    return {}
  }

  if (!id || !secret) {
    throw new Error('Cloudflare Access Client ID and Client Secret must be provided together.')
  }

  return {
    [CF_ACCESS_CLIENT_ID_HEADER]: id,
    [CF_ACCESS_CLIENT_SECRET_HEADER]: secret
  }
}

function hasCloudflareAccessHeaders(headers: RequestHeaders | null | undefined): boolean {
  return Boolean(headers?.[CF_ACCESS_CLIENT_ID_HEADER] && headers?.[CF_ACCESS_CLIENT_SECRET_HEADER])
}

function normalizedHttpProtocol(protocol: string): string {
  if (protocol === 'ws:') {
    return 'http:'
  }

  if (protocol === 'wss:') {
    return 'https:'
  }

  return protocol
}

/** True when a request stays within the configured remote origin + path. */
function requestMatchesRemoteBase(requestUrl: string, remoteBaseUrl: string): boolean {
  let request: URL
  let remote: URL

  try {
    request = new URL(requestUrl)
    remote = new URL(remoteBaseUrl)
  } catch {
    return false
  }

  if (normalizedHttpProtocol(request.protocol) !== normalizedHttpProtocol(remote.protocol)) {
    return false
  }

  if (request.host.toLowerCase() !== remote.host.toLowerCase()) {
    return false
  }

  const basePath = remote.pathname.replace(/\/+$/, '')

  if (!basePath || basePath === '/') {
    return true
  }

  return request.pathname === basePath || request.pathname.startsWith(`${basePath}/`)
}

/** Add the pair without leaving differently-cased duplicate header names. */
function mergeCloudflareAccessHeaders(
  requestHeaders: Record<string, string> = {},
  accessHeaders: RequestHeaders = {}
): Record<string, string> {
  if (!hasCloudflareAccessHeaders(accessHeaders)) {
    return requestHeaders
  }

  const next = { ...requestHeaders }

  const cloudflareNames = new Set([
    CF_ACCESS_CLIENT_ID_HEADER.toLowerCase(),
    CF_ACCESS_CLIENT_SECRET_HEADER.toLowerCase()
  ])

  for (const name of Object.keys(next)) {
    if (cloudflareNames.has(name.toLowerCase())) {
      delete next[name]
    }
  }

  return { ...next, ...accessHeaders }
}

interface RegisteredHeaders {
  expiresAt: number
  headers: RequestHeaders
}

/**
 * Binds a freshly-resolved gateway WS URL to its profile's proxy credential.
 * The exact URL includes the Hermes token/ticket, so simultaneous profiles on
 * the same host can still receive the right Cloudflare service token.
 */
class CloudflareAccessWebSocketHeaderRegistry {
  private readonly entries = new Map<string, RegisteredHeaders>()

  constructor(private readonly ttlMs = 60_000) {}

  remember(wsUrl: string, headers: RequestHeaders, now = Date.now()): void {
    if (!hasCloudflareAccessHeaders(headers)) {
      return
    }

    this.entries.set(new URL(wsUrl).href, { expiresAt: now + this.ttlMs, headers: { ...headers } })
  }

  resolve(wsUrl: string, now = Date.now()): RequestHeaders {
    let key: string

    try {
      key = new URL(wsUrl).href
    } catch {
      return {}
    }

    for (const [url, entry] of this.entries) {
      if (entry.expiresAt <= now) {
        this.entries.delete(url)
      }
    }

    const entry = this.entries.get(key)

    return entry ? { ...entry.headers } : {}
  }
}

export {
  CF_ACCESS_CLIENT_ID_HEADER,
  CF_ACCESS_CLIENT_SECRET_HEADER,
  cloudflareAccessHeaders,
  CloudflareAccessWebSocketHeaderRegistry,
  hasCloudflareAccessHeaders,
  mergeCloudflareAccessHeaders,
  requestMatchesRemoteBase
}
