/**
 * The pool keep-set: which secondary (background-profile) sockets this window
 * must keep open, and the recompute that reaps the rest.
 *
 * Moved out of the useGatewayBoot effect verbatim (#C2a) — nothing here reads
 * hook-local state, only the session stores and the registry's own pins, so it
 * is a plain module rather than a closure. The effect drives it (a 60 s tick
 * plus the store subscriptions); the registry pruner is what it calls.
 */
import { pruneSecondaryGateways } from '@/store/gateway'
import { $activeGatewayProfile, normalizeProfileKey } from '@/store/profile'
import { $activeSessionId, $selectedStoredSessionId, $sessions } from '@/store/session'
import {
  $attentionSessionIds,
  $sessionOwnerHoldRevision,
  $sessionTiles,
  $workingSessionIds,
  foregroundSessionScopes,
  liveSessionScopes,
  openTileGatewayScopes
} from '@/store/session-states'

/**
 * Scopes with a running or needs-input session: registry-scoped
 * (connectionId, profile) keys plus the bare profile of every live local
 * session. Two sources can expose the same profile name (every source has a
 * 'default'), so bare profile names can't represent a non-local source's
 * liveness without keeping the wrong gateway alive. Feeds the pruner's
 * keep-set and the wake path's in-flight-work signal.
 */
export const liveWorkScopes = (): Set<string> => {
  const live = new Set([...$workingSessionIds.get(), ...$attentionSessionIds.get()])
  const scopes = liveSessionScopes()

  for (const session of $sessions.get()) {
    if (live.has(session.id)) {
      scopes.add(normalizeProfileKey(session.profile))
    }
  }

  return scopes
}

/**
 * Bound concurrency cost to consumers: keep a background socket while its
 * profile has a running (working) or blocked (needs-input) session, OR an open
 * owner-routed tile (Bot chats stay on a secondary while chrome stays on the
 * launch profile). Once the last consumer leaves, the socket drops (the gateway
 * behind it keeps running). The active profile is always spared. Do not key this
 * off `entry.retained` — that flag only skips dispose-after-RPC; idle prune is
 * what reclaims hover-warmed sockets after you leave.
 */
export const recomputeKeptGateways = () => {
  const keep = new Set([...liveWorkScopes(), ...foregroundSessionScopes()])

  for (const scope of openTileGatewayScopes()) {
    keep.add(scope)
  }

  // A just-created session's owner hold and every open pane's owner ride in
  // through foregroundSessionScopes above; the registry ALSO reads that set
  // itself (its `foregroundScopes` hook) so the refcount-0 lease releases agree
  // with this pruner. This recompute only has to RUN when they change — see the
  // subscriptions bound below.
  pruneSecondaryGateways(keep)
}

/**
 * Bind the recompute to its clock and to every store that can change the
 * keep-set. The pruner is otherwise event-driven: a socket spared by the
 * min-lifetime grace with no store change afterwards would stay open forever,
 * so the minute tick is its only backstop. Returns one detach.
 */
export const bindKeptGatewayRecompute = (): (() => void) => {
  const tick = setInterval(() => recomputeKeptGateways(), 60_000)

  const detachers = [
    $workingSessionIds.subscribe(recomputeKeptGateways),
    $attentionSessionIds.subscribe(recomputeKeptGateways),
    $activeSessionId.subscribe(recomputeKeptGateways),
    $sessionTiles.subscribe(recomputeKeptGateways),
    $activeGatewayProfile.subscribe(recomputeKeptGateways),
    $selectedStoredSessionId.subscribe(recomputeKeptGateways),
    $sessionOwnerHoldRevision.subscribe(recomputeKeptGateways)
  ]

  return () => {
    clearInterval(tick)

    for (const detach of detachers) {
      detach()
    }
  }
}
