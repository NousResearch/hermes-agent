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

  const headersFor = (requestUrl: string): Record<string, string> => headersByUrl.get(String(requestUrl)) ?? {}

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
