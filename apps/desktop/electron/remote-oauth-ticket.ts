import { makeNousCloudBackendDownError } from './backend-health'
import { gatewayTicketFailure, gatewayTicketTransportMessage } from './connection-config'
import { oauthTicketFailureAuthMessage } from './native-auth-decisions'

interface RemoteOauthTicketDeps {
  hasNativeSession: (baseUrl: string) => boolean
  mintGatewayWsTicket: (baseUrl: string, headers: Record<string, string>) => Promise<string>
  /** Best-effort gateway auth-provider lookup (cached), for failure copy. */
  advertisedAuthProviders?: (baseUrl: string) => Promise<unknown>
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
    // Transport faults keep one headline ("could not reach") but the copy now
    // names the actual failure class — timeout/DNS vs connection refused vs
    // an HTTP fault — so fleet logs stop hiding three different upstream
    // causes behind one sentence (#98647).
    //
    // Failure copy only: probe the gateway's advertised providers so an
    // oauth-mode connection against a password-only backend names the real
    // problem (the auth-mode mismatch) instead of looping the user back
    // into a sign-in that can never persist. Best-effort — any probe
    // failure keeps the existing expired/not-signed-in copy.
    let advertisedProviders: unknown

    if (deps.advertisedAuthProviders) {
      try {
        advertisedProviders = await deps.advertisedAuthProviders(baseUrl)
      } catch {
        advertisedProviders = undefined
      }
    }

    throw (
      makeNousCloudBackendDownError(baseUrl, error) ??
      gatewayTicketFailure(
        error,
        oauthTicketFailureAuthMessage(hadNativeSession, advertisedProviders),
        gatewayTicketTransportMessage(error)
      )
    )
  }
}
