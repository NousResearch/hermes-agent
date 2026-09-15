import { registryGatewayWsUrl } from './plugin-profile-routes'

export interface RegistryGatewayWsConnection {
  authMode: string
  baseUrl: string
  wsUrl: string
  // The id the registry RESOLVED for this request; an empty connectionId in
  // the payload means the primary, so only this one identifies the socket.
  connectionId?: string
  headers?: Record<string, string>
  profile?: null | string
  sharedRemote?: boolean
}

interface RegistryGatewayWsUrlDependencies {
  ensureBackend: (connectionId: unknown, profile: unknown) => Promise<RegistryGatewayWsConnection>
  mintTicket: (baseUrl: string, headers?: Record<string, string>) => Promise<string>
  buildTicketUrl: (baseUrl: string, ticket: string) => string
  // Resolves a payload connectionId the way the backend resolution does: an
  // empty one means the registry's primary. Pooled backends resolve through a
  // shared promise whose connection carries no id, so the key cannot be taken
  // from the connection alone.
  resolveConnectionId?: (connectionId: unknown) => string
  // Receives the resolved connection too, so a cookie-authed gateway can bind
  // its forwarded proxy session to this exact url (see gateway-ws-cookie.ts),
  // plus the consumer key identifying THIS (connectionId, profile) socket: a
  // shared remote serves several profiles at one baseUrl, and each re-mints
  // only its own url.
  rememberHeaders: (
    wsUrl: string,
    headers?: Record<string, string>,
    connection?: RegistryGatewayWsConnection,
    consumer?: string
  ) => Promise<void> | void
}

interface RemoteRequestDetails {
  url: string
  requestHeaders?: Record<string, string>
}

type RemoteRequestCallback = (result: { requestHeaders?: Record<string, string> }) => void

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
    // Pin the source id BEFORE selecting the backend, and select with the
    // pinned id. Resolving it afterwards read the registry's primary a second
    // time: if the primary changed across the await, one request got the old
    // backend under the NEW id, so its next mint no longer recognized -- and
    // so no longer retired -- the url it had just authorized.
    const pinnedId = String(connectionId ?? '').trim() || (dependencies.resolveConnectionId?.(connectionId) ?? '')
    const connection = await dependencies.ensureBackend(pinnedId || connectionId, profile)

    // Distinct from every other pair and from the non-registry mint paths, and
    // normalized as the backend resolution normalizes it: an omitted
    // connectionId means the primary and an omitted profile means 'default',
    // so the same socket re-minting cannot land under a second key and leave
    // its stale ticket url live.
    const resolvedId = pinnedId || String(connection.connectionId ?? '').trim() || 'primary'

    const consumer = `registry:${resolvedId}:${String(profile ?? '').trim() || 'default'}`
    let wsUrl = connection.wsUrl

    if (connection.authMode === 'oauth') {
      const ticket = await dependencies.mintTicket(connection.baseUrl, connection.headers)
      wsUrl = dependencies.buildTicketUrl(connection.baseUrl, ticket)
    }

    const finalWsUrl = registryGatewayWsUrl(connection, wsUrl)

    await dependencies.rememberHeaders(finalWsUrl, connection.headers, connection, consumer)

    return finalWsUrl
  }
}
