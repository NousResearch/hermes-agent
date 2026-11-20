import { RECONNECT_ATTEMPT_TIMEOUT_MS, withTimeout } from '@/lib/with-timeout'

// True when `profile`'s backend route resolves to the SHARED primary backend
// (global-remote case 3 in resolveProfileBackendRoute). Both shared-primary and
// pooled descriptors carry `profile` so WebSocket URL minting targets the right
// profile. `sharedPrimary` is the explicit discriminator; treating every tagged
// descriptor as shared strands local/own-remote pooled profiles on the default
// socket. Dialing a second socket at the shared descriptor is wrong — over SSH
// the second dial fails (tunnel/token are per-backend) and the closed socket
// poisons the active gateway with "not connected" even though the primary is
// open right next to it.
//
// A failed or timed-out lookup REJECTS; it never answers "not shared". On main
// every local profile rides the primary's one multiplexing gateway, and the
// lookup is slow exactly when that gateway is still booting — reading the
// timeout as "not shared" dialed a secondary whose own getConnection joined the
// same boot for another 20 s, then opened a second socket at the shared process
// (#120005). Callers surface the rejection as the switch/request failure.
export async function sharedPrimaryRoute(profile: string): Promise<boolean> {
  const desktop = window.hermesDesktop

  if (!desktop) {
    return false
  }

  // Unbounded IPC round-trip into main (#93454): bound it so a wedge rejects.
  const conn = await withTimeout(
    desktop.getConnection(profile),
    RECONNECT_ATTEMPT_TIMEOUT_MS,
    `Timed out resolving the shared-primary route for profile "${profile}"`
  )

  return conn?.sharedPrimary === true
}

// Scopes whose last route resolution rode the primary socket (shared primary /
// shared remote). A HINT for passive readers only (gateway.ts gatewayRouteState);
// requests still resolve the route per call. Each hint is bound to the primary
// socket + connection it was learned on: after a soft re-home the same profile
// may live on another backend, so a hint from the old primary must not key it
// as `primary:<profile>` under the new one.
export interface PrimaryIdentity {
  connectionId: null | string
  gateway: unknown
}

const primaryRiders = new Map<string, PrimaryIdentity>()

export function notePrimaryRider(owner: PrimaryIdentity, scope: string, rides: boolean): boolean {
  if (rides) {
    primaryRiders.set(scope, owner)
  } else {
    primaryRiders.delete(scope)
  }

  return rides
}

export function isPrimaryRider(scope: string, current: PrimaryIdentity): boolean {
  const owner = primaryRiders.get(scope)

  return Boolean(owner && owner.gateway === current.gateway && owner.connectionId === current.connectionId)
}
