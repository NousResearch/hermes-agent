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
    consumer?: string,
    // False when this caller rewrites the minted url before dialing it, so
    // the url we could authorize is not the one that gets opened.
    authorizeCookie?: boolean
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

// Purposes whose caller REWRITES the minted url before dialing it, so the url
// we could authorize is not the url that gets opened. Speech mints an
// `/api/ws` ticket url and then rewrites the path to `/api/audio/speak-stream`
// (appending its own `profile` param), so authorizing the minted url would
// park a credential no upgrade can ever consume while the speech upgrade
// still carries none. Forwarding onto that endpoint is deliberately out of
// scope; the single-use ticket in its url is what authenticates it to Hermes.
const WS_PURPOSES_THAT_REWRITE_THE_URL = new Set(['speech'])

// Renderer-supplied, so normalized once, here, before it can reach a map key
// or a scope decision.
function normalizeWsPurpose(purpose: unknown) {
  return String(purpose ?? '')
    .trim()
    .slice(0, 32)
}

export function gatewayWsAuthorizesItsMintedUrl(purpose?: unknown) {
  return !WS_PURPOSES_THAT_REWRITE_THE_URL.has(normalizeWsPurpose(purpose))
}

// Which socket consumer is asking: the calling window -- senders are distinct
// per window, including session and peer windows -- plus what it opens, since
// one window's chat, speech and secondary flows mint against the same route
// for independent sockets. Stable across that consumer's reconnects, so its
// own re-mint still retires its own url.
export function gatewayWsConsumerTag(event?: { sender?: { id?: unknown } }, purpose?: unknown) {
  const sender = event?.sender?.id
  const kind = normalizeWsPurpose(purpose)

  return `w${typeof sender === 'number' ? sender : 0}:${kind || 'default'}`
}

export function createRegistryGatewayWsUrlHandler(dependencies: RegistryGatewayWsUrlDependencies) {
  // `consumerTag` identifies the actual socket consumer behind the route --
  // which window asked, and what it opens (chat vs speech). A route is not a
  // socket: two windows, or one window's chat and speech flows, mint against
  // the same (connectionId, profile) and hold independent pending upgrades.
  // Without it, the second mint retires the first's unused authorization and
  // that upgrade goes out without its proxy cookie.
  return async (payload: unknown, consumerTag?: string, authorizeCookie = true): Promise<string> => {
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

    const consumer = `registry:${resolvedId}:${String(profile ?? '').trim() || 'default'}:${consumerTag || 'default'}`
    let wsUrl = connection.wsUrl

    if (connection.authMode === 'oauth') {
      const ticket = await dependencies.mintTicket(connection.baseUrl, connection.headers)
      wsUrl = dependencies.buildTicketUrl(connection.baseUrl, ticket)
    }

    const finalWsUrl = registryGatewayWsUrl(connection, wsUrl)

    await dependencies.rememberHeaders(finalWsUrl, connection.headers, connection, consumer, authorizeCookie)

    return finalWsUrl
  }
}
