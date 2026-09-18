import { makeNousCloudBackendDownError } from './backend-health'
import { gatewayTicketFailure } from './connection-config'
import { oauthTicketFailureAuthMessage } from './native-auth-decisions'

interface RemoteOauthTicketDeps {
  hasNativeSession: (baseUrl: string) => boolean
  mintGatewayWsTicket: (baseUrl: string, headers: Record<string, string>) => Promise<string>
}

// Roster dials use this same mint path before readiness; the ordinary 10s
// deadline can discard a healthy OAuth source while its cold session warms.
export function rosterSourceEnumerationTimeoutMs(connection: { kind?: string; authMode?: string }): number {
  return (connection.kind === 'remote' || connection.kind === 'cloud') && connection.authMode === 'oauth'
    ? 30_000
    : 10_000
}

export async function resolveRemoteOauthTicket(
  baseUrl: string,
  headers: Record<string, string>,
  deps: RemoteOauthTicketDeps
): Promise<string> {
  // The mint is authoritative: a cold cookie partition may not yet report a
  // session. Snapshot native state only for copy; a rejected mint can erase it.
  const hadNativeSession = deps.hasNativeSession(baseUrl)

  try {
    return await deps.mintGatewayWsTicket(baseUrl, headers)
  } catch (error) {
    throw (
      makeNousCloudBackendDownError(baseUrl, error) ??
      gatewayTicketFailure(
        error,
        oauthTicketFailureAuthMessage(hadNativeSession),
        'Could not reach the remote Hermes gateway while refreshing its WebSocket ticket. Try reconnecting.'
      )
    )
  }
}
