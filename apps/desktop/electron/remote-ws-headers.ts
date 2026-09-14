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
    onBeforeSendHeaders?: (
      listener: (details: RemoteRequestDetails, callback: RemoteRequestCallback) => void
    ) => void
  }
}

/** Strip CR/LF that Windows clipboard pastes leave on service-token secrets. */
export function sanitizeRemoteHeaderValue(value: string): string {
  return String(value || '')
    .replace(/[\r\n]+/g, '')
    .trim()
}

export function sanitizeRemoteHeaderMap(headers: Record<string, string> = {}): Record<string, string> {
  const out: Record<string, string> = {}

  for (const [name, value] of Object.entries(headers)) {
    const clean = sanitizeRemoteHeaderValue(value)

    if (name && clean) {
      out[name] = clean
    }
  }

  return out
}

/**
 * Header blocks that Chromium (login window, renderer WS) may attach to a
 * remote gateway request. Registry Connections are the live source; the v1
 * single-remote block is fallback. Longer base URLs win so a path-prefixed
 * gateway is not shadowed by its origin sibling.
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

    sources.push({ headers: sanitizeRemoteHeaderMap(connection.headers), url: connection.url })
  }

  if (input.v1Remote?.url && input.v1Remote.headers && Object.keys(input.v1Remote.headers).length > 0) {
    sources.push({
      headers: sanitizeRemoteHeaderMap(input.v1Remote.headers),
      url: input.v1Remote.url
    })
  }

  return sources.sort((a, b) => String(b.url || '').length - String(a.url || '').length)
}

export function resolveRemoteRequestHeaders(
  requestUrl: string,
  options: { exactHeaders?: Record<string, string>; sources?: RemoteHeaderSource[] } = {}
): Record<string, string> {
  const exact = sanitizeRemoteHeaderMap(options.exactHeaders || {})

  if (Object.keys(exact).length > 0) {
    return exact
  }

  for (const source of options.sources || []) {
    if (!source.url || !source.headers) {
      continue
    }

    const headers = sanitizeRemoteHeaderMap(source.headers)

    if (Object.keys(headers).length > 0 && remoteRequestMatchesBaseUrl(requestUrl, source.url)) {
      return headers
    }
  }

  return {}
}

export function formatLoadUrlExtraHeaders(headers: Record<string, string> = {}): string {
  return Object.entries(sanitizeRemoteHeaderMap(headers))
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
    if (!wsUrl || Object.keys(headers).length === 0) {
      return
    }

    headersByUrl.set(String(wsUrl), headers)

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

  callback({ requestHeaders: { ...details.requestHeaders, ...headers } })
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
