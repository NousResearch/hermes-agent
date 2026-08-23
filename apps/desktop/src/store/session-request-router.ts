import {
  activeGatewayConnectionId,
  activeGatewayProfileKey,
  requestGatewayForAgent,
  requestGatewayForProfile
} from '@/store/gateway'

export interface SessionProfileRoute {
  connectionId: string
  mode?: 'local' | 'remote'
  profile: string
  targetProfile?: string
}

export type SessionOwnerScope = undefined | null | string | SessionProfileRoute

// ── Session-scoped RPC routing (the #89206 class) ───────────────────────────
// A session-scoped RPC (session.resume / session.activate / session.usage)
// only means anything on the backend that OWNS the session's profile. The
// ambient "active gateway" is a moving target: between the profile-swap await
// and the RPC dispatch, a concurrent switch, an idle-reap eviction, a failed
// dial, or a connection edit can re-point the active route at another
// backend. Dispatching on it anyway lands the RPC on a backend that has never
// heard of the session — it 404s or times out, the renderer burns its bounded
// retries, and the user sees "retries gave up" while the session's own
// backend is healthy (blank Bot Chats, dead wake-ups; local pool and SSH
// alike). These helpers make the owning profile, resolved at REQUEST time,
// the routing authority.

const normKey = (profile: null | string | undefined): string => (profile ?? '').trim() || 'default'

const isRoute = (owner: SessionOwnerScope): owner is SessionProfileRoute =>
  Boolean(owner && typeof owner === 'object' && 'connectionId' in owner)

function routeParams(route: SessionProfileRoute, params: Record<string, unknown>): Record<string, unknown> {
  if (!route.targetProfile || !Object.prototype.hasOwnProperty.call(params, 'profile')) {
    return params
  }

  return { ...params, profile: route.targetProfile }
}

/**
 * True when a session-scoped RPC must be pinned to `ownerProfile`'s own
 * socket because the active gateway currently serves a different profile.
 * A null/empty owner means the session's profile is unknown — route ambient
 * (the pre-multi-profile behavior) rather than guessing.
 */
export function sessionRpcNeedsProfileRoute(
  ownerProfile: SessionOwnerScope | undefined,
  activeProfile: string = activeGatewayProfileKey(),
  activeConnectionId: null | string = activeGatewayConnectionId()
): boolean {
  if (isRoute(ownerProfile)) {
    // A descriptor is an immutable ownership claim. Even an explicitly local
    // route must not collapse to the ambient request: another connection can
    // expose the same profile name, and activation is UI state only.
    return Boolean(ownerProfile.connectionId.trim())
  }

  if (ownerProfile == null || !String(ownerProfile).trim()) {
    return false
  }

  return normKey(ownerProfile) !== normKey(activeProfile)
}

/**
 * Dispatch a session-scoped RPC on the socket that owns `ownerProfile`,
 * falling back to the ambient dispatcher when the active gateway already
 * serves that profile (keeps the primary's reauth-aware reconnect path).
 * The route is decided at CALL time, not at swap time.
 */
export function requestForSessionProfile<T>(
  ownerProfile: SessionOwnerScope | undefined,
  ambientRequest: <R>(
    method: string,
    params?: Record<string, unknown>,
    timeoutMs?: number,
    signal?: AbortSignal
  ) => Promise<R>,
  method: string,
  params: Record<string, unknown> = {},
  timeoutMs?: number,
  signal?: AbortSignal
): Promise<T> {
  if (isRoute(ownerProfile)) {
    const connectionId = ownerProfile.connectionId.trim()

    if (!connectionId) {
      return Promise.reject(new Error('Session owner route is missing connectionId'))
    }

    const routedParams = routeParams(ownerProfile, params)

    return timeoutMs === undefined && signal === undefined
      ? requestGatewayForAgent<T>(connectionId, normKey(ownerProfile.profile), method, routedParams)
      : requestGatewayForAgent<T>(connectionId, normKey(ownerProfile.profile), method, routedParams, timeoutMs, signal)
  }

  if (!sessionRpcNeedsProfileRoute(ownerProfile)) {
    // Forward the extra args only when the caller actually supplied them. The
    // ambient dispatcher is a plain gateway request whose arity callers assert
    // on; handing it a trailing `undefined, undefined` on every session RPC
    // changes the observed call shape for the many callers that never asked
    // for a deadline (the plugin host bridge in contrib/wiring is the only one
    // that does).
    return timeoutMs === undefined && signal === undefined
      ? ambientRequest<T>(method, params)
      : ambientRequest<T>(method, params, timeoutMs, signal)
  }

  return requestGatewayForProfile<T>(normKey(ownerProfile), method, params, timeoutMs, signal)
}
