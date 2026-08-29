import { atom } from 'nanostores'

import type { UnreadSessionTarget } from './session-dot-state'
import type { SessionProfileRoute } from './session-request-router'

/** Navigation intent emitted by pane chrome and consumed by the mounted
 * Sessions wiring. State survives StrictMode effect teardown/re-setup, unlike
 * an imperative handler stored in a module variable. */
export const $openNextUnreadRequest = atom(0)

export function requestOpenNextUnread(): void {
  $openNextUnreadRequest.set($openNextUnreadRequest.get() + 1)
}

export type UnreadTargetProfileScope = SessionProfileRoute | string | undefined

export interface UnreadTargetPreflightCandidate {
  ownerRoute: null | SessionProfileRoute
  scope: UnreadTargetProfileScope
}

/** Route to persist before opening a connection-owned target. */
export function ownerRouteForUnreadTarget(
  target: UnreadSessionTarget,
  activeConnectionId?: null | string,
  registeredRoutes: readonly SessionProfileRoute[] = []
): SessionProfileRoute | null {
  // Unified-list rows from secondary gateways carry connectionId. Rows served
  // by the currently-primary registered gateway do not, so freeze the active
  // source as their owner before navigation can move presentation elsewhere.
  const connectionId = target.connectionId?.trim() || activeConnectionId?.trim()
  const profile = target.profile?.trim() || 'default'

  if (!connectionId) {
    return null
  }

  const route = registeredRoutes.find(
    candidate => candidate.connectionId.trim() === connectionId && candidate.profile.trim() === profile
  )

  return route
    ? {
        connectionId,
        ...(route.mode ? { mode: route.mode } : {}),
        profile,
        ...(route.targetProfile?.trim() ? { targetProfile: route.targetProfile.trim() } : {})
      }
    : { connectionId, profile }
}

/** Exact REST scope for a target. A registered Desktop profile can map to a
 * differently named backend profile, so use targetProfile when the registry
 * route provides one. */
export function profileScopeForUnreadTarget(
  target: UnreadSessionTarget,
  activeConnectionId?: null | string,
  registeredRoutes: readonly SessionProfileRoute[] = []
): UnreadTargetProfileScope {
  const ownerRoute = ownerRouteForUnreadTarget(target, activeConnectionId, registeredRoutes)

  if (ownerRoute) {
    return { connectionId: ownerRoute.connectionId, profile: ownerRoute.targetProfile || ownerRoute.profile }
  }

  return target.profile?.trim() || undefined
}

/** Ordered owner probes for an unread row. Primary rows intentionally omit a
 * connection id: profile-only routing must get the first chance so Electron can
 * honor per-profile remote overrides. Registry scoping is the fallback for a
 * global-remote primary and the only path for explicitly tagged secondary rows. */
export function preflightCandidatesForUnreadTarget(
  target: UnreadSessionTarget,
  activeConnectionId?: null | string,
  registeredRoutes: readonly SessionProfileRoute[] = []
): UnreadTargetPreflightCandidate[] {
  const candidates: UnreadTargetPreflightCandidate[] = []

  if (!target.connectionId?.trim()) {
    candidates.push({ ownerRoute: null, scope: target.profile?.trim() || undefined })
  }

  const ownerRoute = ownerRouteForUnreadTarget(target, activeConnectionId, registeredRoutes)

  if (ownerRoute) {
    candidates.push({
      ownerRoute,
      scope: { connectionId: ownerRoute.connectionId, profile: ownerRoute.targetProfile || ownerRoute.profile }
    })
  }

  return candidates.length ? candidates : [{ ownerRoute: null, scope: undefined }]
}

export type PreflightUnreadTarget = (
  target: UnreadSessionTarget
) => Promise<null | SessionProfileRoute>

/** Open the first target whose row still resolves on its owning backend. A
 * rejected preflight leaves that candidate unread and advances to the next. */
export async function openNextValidUnread(
  targets: readonly UnreadSessionTarget[],
  preflight: PreflightUnreadTarget,
  open: (target: UnreadSessionTarget, ownerRoute: null | SessionProfileRoute) => void
): Promise<UnreadSessionTarget | null> {
  for (const target of targets) {
    let ownerRoute: null | SessionProfileRoute

    try {
      ownerRoute = await preflight(target)
    } catch {
      // Stale or unavailable rows remain unread; try the next candidate.
      continue
    }

    open(target, ownerRoute)

    return target
  }

  return null
}
