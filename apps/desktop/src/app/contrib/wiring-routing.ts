/**
 * Pure routing helpers for the contrib wiring controller.
 *
 * Kept out of wiring.tsx so they can be unit-tested without importing the whole
 * React/Electron controller module.
 */

/**
 * Resolve a runtime session id back to its stored id by reverse-scanning the
 * stored->runtime binding map — the same ladder use-session-tile-delegate's
 * `storedSessionIdForRuntime` uses. Returns undefined when the id isn't a known
 * runtime id, so the caller can treat it as already a stored id.
 */
export function findStoredIdForRuntimeId(bindings: Map<string, string>, runtimeId: string): string | undefined {
  for (const [storedId, mapped] of bindings) {
    if (mapped === runtimeId) {
      return storedId
    }
  }

  return undefined
}

/**
 * The stored session id a session-scoped RPC should route by.
 *
 * Route by the session the RPC TARGETS (its `session_id` param), not by the
 * window's focused tile: `requestGateway` is one shared closure for every
 * session RPC, so keying off the focused tile sent a non-focused tile's RPC
 * (a bot chat while another pane is active) to the focused tile's backend — the
 * Bot Mode misroute. `session_id` is a RUNTIME id while tiles/rows key on the
 * STORED id, so translate via the state cache, then the reverse binding scan;
 * an unknown id is already a stored id (several RPCs pass stored ids directly).
 * With no `session_id` at all (ambient/config calls) fall back to the focused
 * then selected tile.
 */
export function resolveRoutingSessionId(args: {
  paramSessionId: string | undefined
  storedIdForRuntime: (runtimeId: string) => string | undefined
  focusedStoredSessionId: null | string
  selectedStoredSessionId: null | string
}): null | string {
  const { focusedStoredSessionId, paramSessionId, selectedStoredSessionId, storedIdForRuntime } = args

  if (paramSessionId) {
    return storedIdForRuntime(paramSessionId) ?? paramSessionId
  }

  return focusedStoredSessionId ?? selectedStoredSessionId
}

/** The owner shapes the ladder below can return: an exact route (connection +
 *  profile), a bare profile name, or undefined (unknown — probe, never
 *  "active"). Structural twin of store/session-request-router's
 *  SessionOwnerScope, kept local so this module stays import-free. */
export interface SessionRpcOwnerRoute {
  connectionId: string
  mode?: 'local' | 'remote'
  profile: string
  targetProfile?: string
}

/**
 * The SYNC owner a session-scoped RPC routes to, resolved in this order:
 *
 *   1. the persisted tile owner route (a bot chat / split tile records the
 *      exact connectionId + profile it was opened with, survives relaunch);
 *   2. the exact, UNIQUE session owner hint (recorded the moment a routed
 *      session.create returns, or at plugin open time) — the only durable
 *      exact owner a fresh main-pane chat or a hidden session has;
 *   3. the session row's owner — an EXACT route when the row is
 *      connection-tagged (optimistic row from a routed create, or the
 *      unified list splice), else its bare profile (the cross-profile
 *      aggregator tags rows, but a bare profile loses the connection and
 *      can lag the create);
 *   4. undefined → the caller runs the cross-profile probe.
 *
 * The hint outranks the row because the row is presentation state that can
 * be stamped from the AMBIENT profile (an optimistic row minted while
 * All-profiles / Bot routing left `default` active), and because it carries
 * no connection: a fresh chat created on `local::omar` whose row read
 * `default` ran its first turn on omar and then 4001'd "session not found"
 * on the second, when the row's `default` owner won the route.
 */
export function resolveSessionRpcOwner(args: {
  routingSessionId: null | string
  tileOwnerRoute: (storedSessionId: string) => SessionRpcOwnerRoute | undefined
  sessionOwnerHint: (storedSessionId: string) => SessionRpcOwnerRoute | undefined
  sessionRowOwner: (storedSessionId: string) => null | SessionRpcOwnerRoute | string | undefined
}): SessionRpcOwnerRoute | string | undefined {
  const { routingSessionId, sessionOwnerHint, sessionRowOwner, tileOwnerRoute } = args

  if (!routingSessionId) {
    return undefined
  }

  const fromRow = sessionRowOwner(routingSessionId)

  return (
    tileOwnerRoute(routingSessionId) ??
    sessionOwnerHint(routingSessionId) ??
    (typeof fromRow === 'string' ? fromRow.trim() || undefined : (fromRow ?? undefined))
  )
}
