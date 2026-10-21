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
  id?: number
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
    onCompleted?: (listener: (details: { id?: number }) => void) => void
    onErrorOccurred?: (listener: (details: { id?: number }) => void) => void
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
  const exact = options.exactHeaders || {}

  if (Object.keys(exact).length > 0) {
    return exact
  }

  for (const source of options.sources || []) {
    if (!source.url || !source.headers) {
      continue
    }

    if (Object.keys(source.headers).length > 0 && remoteRequestMatchesBaseUrl(requestUrl, source.url)) {
      return source.headers
    }
  }

  return {}
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
  // Chromium carries app-set headers across redirects, so a configured secret
  // injected into one hop rides along to the next. Track the header names we
  // injected per request id: an out-of-scope hop of THAT request gets them
  // stripped, while an unrelated request carrying the same name (even the
  // same value) is left alone.
  const injectedByRequest: InjectedHeadersByRequest = new Map()

  sessionLike?.webRequest?.onBeforeSendHeaders?.((details, callback) => {
    applyRemoteRequestHeaders(details, callback, headersForRequest, injectedByRequest)
  })

  const forget = (details: { id?: number }) => {
    if (details?.id !== undefined) {
      injectedByRequest.delete(details.id)
    }
  }

  sessionLike?.webRequest?.onCompleted?.(forget)
  sessionLike?.webRequest?.onErrorOccurred?.(forget)
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

/** Lower-cased header names this session injected, keyed by webRequest id. */
export type InjectedHeadersByRequest = Map<number, Set<string>>

export function applyRemoteRequestHeaders(
  details: RemoteRequestDetails,
  callback: RemoteRequestCallback,
  headersForRequest: (requestUrl: string) => Record<string, string>,
  injectedByRequest: InjectedHeadersByRequest
) {
  const headers = headersForRequest(details.url)
  const headerEntries = Object.entries(headers)
  const tracked = details.id === undefined ? undefined : injectedByRequest.get(details.id)

  if (headerEntries.length === 0 && !tracked) {
    callback({})

    return
  }

  // Names to drop from the outgoing request: whatever an earlier hop of this
  // request had injected (it may be out of scope now) plus the names we are
  // about to set (so a differently-cased copy does not survive alongside).
  const strip = new Set(tracked)
  const requestHeaders: Record<string, string> = {}

  for (const [name, value] of headerEntries) {
    strip.add(name.toLowerCase())
    requestHeaders[name] = value
  }

  const outgoing: Record<string, string> = {}

  for (const [name, value] of Object.entries(details.requestHeaders || {})) {
    if (!strip.has(name.toLowerCase())) {
      outgoing[name] = value
    }
  }

  if (details.id !== undefined) {
    if (headerEntries.length > 0) {
      injectedByRequest.set(details.id, new Set(headerEntries.map(([name]) => name.toLowerCase())))
    } else {
      injectedByRequest.delete(details.id)
    }
  }

  callback({ requestHeaders: { ...outgoing, ...requestHeaders } })
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
