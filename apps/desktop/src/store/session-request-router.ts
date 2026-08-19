import { activeGatewayProfileKey, requestGatewayForProfile } from '@/store/gateway'

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

/**
 * True when a session-scoped RPC must be pinned to `ownerProfile`'s own
 * socket because the active gateway currently serves a different profile.
 * A null/empty owner means the session's profile is unknown — route ambient
 * (the pre-multi-profile behavior) rather than guessing.
 */
export function sessionRpcNeedsProfileRoute(
  ownerProfile: null | string | undefined,
  activeProfile: string = activeGatewayProfileKey()
): boolean {
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
  ownerProfile: null | string | undefined,
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
  if (!sessionRpcNeedsProfileRoute(ownerProfile)) {
    return ambientRequest<T>(method, params, timeoutMs, signal)
  }

  return requestGatewayForProfile<T>(normKey(ownerProfile), method, params, timeoutMs, signal)
}
