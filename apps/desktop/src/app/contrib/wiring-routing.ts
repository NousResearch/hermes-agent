/**
 * Pure routing helpers for the contrib wiring controller.
 *
 * Kept out of wiring.tsx so they can be unit-tested without importing the whole
 * React/Electron controller module.
 */

import type { ReplayGapSource } from '@/store/gateway'
import type { SessionOwnerRoute } from '@/store/session-request-router'

export function resolveReplayHydrationTarget<
  T extends { connection_id?: null | string; profile?: null | string }
>(candidates: T[], source: ReplayGapSource): { row: T; scope: { connectionId: null | string; profile: string } } | undefined {
  const sourceProfile = (source.profile ?? '').trim() || 'default'

  const profileCandidates = candidates.filter(candidate => {
    const ownerProfile = (candidate.profile ?? '').trim() || 'default'

    return ownerProfile === sourceProfile
  })

  const row =
    profileCandidates.find(candidate => (candidate.connection_id ?? '').trim() === source.connectionId) ??
    profileCandidates.find(candidate => {
      const ownerConnectionId = (candidate.connection_id ?? '').trim()

      return !ownerConnectionId
    })

  if (!row) {
    return undefined
  }

  return {
    row,
    scope: { connectionId: source.connectionId, profile: sourceProfile }
  }
}

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
 *  "active"). The type-only import keeps this module runtime-import-free. */
export type SessionRpcOwnerRoute = SessionOwnerRoute

/**
 * The SYNC owner a session-scoped RPC routes to, resolved in this order:
 *
 *   1. the persisted tile owner route (a bot chat / split tile records the
 *      exact connectionId + profile it was opened with, survives relaunch);
 *   2. the exact, UNIQUE session owner hint (recorded the moment a routed
 *      session.create returns, or at plugin open time; persisted, bounded);
 *   3. the session row's owner — an EXACT route when the row is
 *      connection-tagged (optimistic row from a routed create, the unified
 *      list splice, or a tag carried across a refresh), else its bare
 *      profile (the cross-profile aggregator tags rows, but a bare profile
 *      loses the connection and can lag the create);
 *   4. undefined → the caller runs the cross-profile probe, and fails closed
 *      if that misses too.
 *
 * The hint outranks the row because the row is presentation state that can
 * be stamped from the AMBIENT profile (an optimistic row minted while
 * All-profiles / Bot routing left `default` active), and because it carries
 * no connection: a fresh chat created on `local::omar` whose row read
 * `default` ran its first turn on omar and then 4001'd "session not found"
 * on the second, when the row's `default` owner won the route. The
 * connection-tagged row rung is what keeps two-turn continuity from resting
 * on the transient hint alone (bounded, evictable, gone after a relaunch).
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
