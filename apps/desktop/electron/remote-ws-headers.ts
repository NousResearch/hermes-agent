import { remoteRequestMatchesBaseUrl } from './connection-config'
import { registryGatewayWsUrl } from './plugin-profile-routes'

export interface RegistryGatewayWsConnection {
  authMode: string
  baseUrl: string
  wsUrl: string
  headers?: Record<string, string>
  profile?: null | string
  sharedRemote?: boolean
}

interface RegistryGatewayWsUrlDependencies {
  ensureBackend: (connectionId: unknown, profile: unknown) => Promise<RegistryGatewayWsConnection>
  mintTicket: (baseUrl: string, headers?: Record<string, string>) => Promise<string>
  buildTicketUrl: (baseUrl: string, ticket: string) => string
  rememberHeaders: (wsUrl: string, headers?: Record<string, string>) => void
}

interface RemoteRequestDetails {
  url: string
  requestHeaders?: Record<string, string>
}

type RemoteRequestCallback = (result: { requestHeaders?: Record<string, string> }) => void

export interface RemoteHeaderSource {
  headers?: Record<string, string>
  kind?: string
  url?: string
}

interface SessionLike {
  webRequest?: {
    onBeforeSendHeaders?: (listener: (details: RemoteRequestDetails, callback: RemoteRequestCallback) => void) => void
  }
}

/**
 * Header blocks that Chromium (login window, renderer WS) may attach to a
 * remote gateway request. Registry Connections are the live source; the v1
 * single-remote block is fallback. Longer base URLs win so a path-prefixed
 * gateway is not shadowed by its origin sibling.
 *
 * Values arrive already sanitized: decryptRemoteHeaders in main.ts strips
 * CR/LF on the single decrypt funnel, so there is no second normalizer here.
 */
export function collectRemoteHeaderSources(input: {
  connections?: RemoteHeaderSource[]
  v1Remote?: null | RemoteHeaderSource
}): RemoteHeaderSource[] {
  const sources: RemoteHeaderSource[] = []

  for (const connection of input.connections || []) {
    if (connection.kind && connection.kind !== 'remote' && connection.kind !== 'cloud') {
      continue
    }

    if (!connection.url || !connection.headers || Object.keys(connection.headers).length === 0) {
      continue
    }

    sources.push({ headers: connection.headers, url: connection.url })
  }

  if (input.v1Remote?.url && input.v1Remote.headers && Object.keys(input.v1Remote.headers).length > 0) {
    sources.push({
      headers: input.v1Remote.headers,
      url: input.v1Remote.url
    })
  }

  return sources.sort((a, b) => String(b.url || '').length - String(a.url || '').length)
}

export function resolveRemoteRequestHeaders(
  requestUrl: string,
  options: { exactHeaders?: Record<string, string>; sources?: RemoteHeaderSource[] } = {}
): Record<string, string> {
  const { Origin, ...exact } = options.exactHeaders || {}

  if (Object.keys(exact).length > 0) {
    return Origin ? { ...exact, Origin } : exact
  }

  for (const source of options.sources || []) {
    if (!source.url || !source.headers) {
      continue
    }

    if (Object.keys(source.headers).length > 0 && remoteRequestMatchesBaseUrl(requestUrl, source.url)) {
      return Origin ? { ...source.headers, Origin } : source.headers
    }
  }

  return Origin ? { Origin } : {}
}

export function formatLoadUrlExtraHeaders(headers: Record<string, string> = {}): string {
  return Object.entries(headers)
    .map(([name, value]) => `${name}: ${value}`)
    .join('\n')
}

export function oauthLoginLoadUrlOptions(headers: Record<string, string> = {}): { extraHeaders?: string } {
  const extraHeaders = formatLoadUrlExtraHeaders(headers)

  return extraHeaders ? { extraHeaders } : {}
}

export function attachRemoteRequestHeaderListener(
  sessionLike: SessionLike,
  headersForRequest: (requestUrl: string) => Record<string, string>
) {
  sessionLike?.webRequest?.onBeforeSendHeaders?.((details, callback) => {
    applyRemoteRequestHeaders(details, callback, headersForRequest)
  })
}

export function createRemoteWsHeaderStore(limit = 100) {
  const headersByUrl = new Map<string, Record<string, string>>()

  const remember = (wsUrl: string, headers: Record<string, string> = {}) => {
    let url: URL

    try {
      url = new URL(wsUrl)
    } catch {
      return
    }

    if (!['ws:', 'wss:'].includes(url.protocol)) {
      return
    }

    // Main vouches only for exact authenticated gateway URLs. Native renderers
    // use loopback HTTP in both dev and packaged builds; a non-web Origin avoids
    // depending on a reverse proxy's public Host or Host rewriting policy.
    headersByUrl.set(String(wsUrl), { ...headers, Origin: 'null' })

    while (headersByUrl.size > limit) {
      const oldest = headersByUrl.keys().next().value

      if (!oldest) {
        break
      }

      headersByUrl.delete(oldest)
    }
  }

  const headersFor = (requestUrl: string): Record<string, string> => {
    const key = String(requestUrl)
    const headers = headersByUrl.get(key)

    if (!headers) {
      return {}
    }

    headersByUrl.delete(key)
    headersByUrl.set(key, headers)

    return headers
  }

  return { headersFor, remember }
}

export function applyRemoteRequestHeaders(
  details: RemoteRequestDetails,
  callback: RemoteRequestCallback,
  headersForRequest: (requestUrl: string) => Record<string, string>
) {
  const headers = headersForRequest(details.url)

  if (Object.keys(headers).length === 0) {
    callback({})

    return
  }

  const requestHeaders = { ...details.requestHeaders, ...headers }

  if (/^wss?:/.test(details.url) && headers.Origin) {
    const originalOrigin = Object.entries(details.requestHeaders || {}).find(
      ([name]) => name.toLowerCase() === 'origin'
    )?.[1]

    let nativeOrigin = !originalOrigin || originalOrigin === 'null'

    try {
      const source = new URL(originalOrigin || '')
      nativeOrigin ||=
        ['file:', 'app:'].includes(source.protocol) ||
        (['http:', 'https:'].includes(source.protocol) && ['localhost', '127.0.0.1', '[::1]'].includes(source.hostname))
    } catch {
      // Malformed origins are not vouched for as desktop renderer origins.
    }

    for (const name of Object.keys(requestHeaders)) {
      if (name.toLowerCase() === 'origin') {
        delete requestHeaders[name]
      }
    }

    // Don't let an untrusted web frame borrow the native Origin stamp. Its
    // original origin must still pass the remote dashboard's unchanged guard.
    if (nativeOrigin) {
      requestHeaders.Origin = headers.Origin
    } else if (originalOrigin) {
      requestHeaders.Origin = originalOrigin
    }
  }

  callback({ requestHeaders })
}

export function createRegistryGatewayWsUrlHandler(dependencies: RegistryGatewayWsUrlDependencies) {
  return async (payload: unknown): Promise<string> => {
    const { connectionId, profile } = payload && typeof payload === 'object' ? (payload as any) : ({} as any)
    const connection = await dependencies.ensureBackend(connectionId, profile)
    let wsUrl = connection.wsUrl

    if (connection.authMode === 'oauth') {
      const ticket = await dependencies.mintTicket(connection.baseUrl, connection.headers)
      wsUrl = dependencies.buildTicketUrl(connection.baseUrl, ticket)
    }

    const finalWsUrl = registryGatewayWsUrl(connection, wsUrl)

    dependencies.rememberHeaders(finalWsUrl, connection.headers)

    return finalWsUrl
  }
}
