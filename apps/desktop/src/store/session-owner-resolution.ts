/**
 * Fail-closed owner resolution for session-scoped RPCs.
 *
 * A request that carries a `session_id` only means anything on the backend
 * that OWNS that session. When every rung of the owner ladder (tile route →
 * exact owner hint → connection-tagged / profiled row → cross-profile REST
 * probe) misses, the request must NOT quietly ride the ambient presentation
 * gateway: "active" is presentation state with no routing authority, and an
 * ambient fallback turns missing ownership metadata into a misleading backend
 * "session not found" (or, worse, an answer from a backend that merely happens
 * to know a same-named session). Surface an explicit owner-resolution error
 * instead — the caller's error UX shows it, and the runtime that minted the
 * session is left untouched for the next correctly-routed attempt.
 *
 * The ONE case where the ambient gateway is not a fallback but the owner by
 * construction: no registry topology exists (legacy v1 primary) AND at most
 * one profile exists — a single backend serves every session, so there is
 * nothing to misroute to. Older single-profile backends omit `profile` on
 * their rows entirely; those users keep working unchanged.
 */
import { hasRegistryTopology } from './connection-registry-state'
import { $profiles } from './profile'
import { isSessionOwnerRoute, type SessionOwnerScope } from './session-request-router'

export class SessionOwnerResolutionError extends Error {
  constructor(
    readonly sessionId: string,
    readonly method: string
  ) {
    super(
      `Session owner could not be resolved for "${sessionId}" (${method}): ` +
        'no owner route, hint, connection-tagged row or profile probe named the backend that holds this session, ' +
        'and routing it to the active gateway would be a guess.'
    )
    this.name = 'SessionOwnerResolutionError'
  }
}

export function isSessionOwnerResolutionError(error: unknown): error is SessionOwnerResolutionError {
  return (
    error instanceof SessionOwnerResolutionError ||
    (error as { name?: unknown })?.name === 'SessionOwnerResolutionError'
  )
}

/** True when the ambient gateway is provably the only backend any session
 *  can live on (legacy single-backend Desktop): Electron has published no
 *  connection registry and there is at most one profile. The active route is
 *  presentation state; a null active connection does not prove sole topology. */
export function ambientGatewayOwnsEverySession(): boolean {
  return !hasRegistryTopology() && $profiles.get().length <= 1
}

/** True when `owner` names a backend: an exact connection route, or a bare
 *  profile. A bare profile stays an owner in registry topology too — a profile
 *  pick on the primary or the explicit `local` source takes the legacy
 *  profile-only door (store/profile activateOnCurrentSource, so a per-profile
 *  remote override resolves), and a session minted there is owned by that
 *  profile's pool socket, which requestForSessionProfile dials by name. */
export function sessionOwnerIsKnown(owner: SessionOwnerScope): boolean {
  if (isSessionOwnerRoute(owner)) {
    return Boolean(owner.connectionId.trim())
  }

  return owner != null && Boolean(String(owner).trim())
}

/**
 * Gate before a session-scoped RPC falls to the ambient dispatcher. Throws
 * SessionOwnerResolutionError when the session's owner is unknown and the
 * ambient gateway is not the sole backend; otherwise returns normally.
 */
export function assertSessionOwnerResolved(
  owner: SessionOwnerScope,
  context: { method: string; sessionId: null | string | undefined }
): void {
  if (!context.sessionId || sessionOwnerIsKnown(owner) || ambientGatewayOwnsEverySession()) {
    return
  }

  throw new SessionOwnerResolutionError(context.sessionId, context.method)
}
