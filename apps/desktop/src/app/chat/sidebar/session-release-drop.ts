import type { SessionDragPayload } from '@/app/chat/composer/inline-refs'

import { placeSessionIdAtAnchor, type SessionDropAnchor } from './use-session-drop-zone'

export type SidebarSessionReleaseSection = 'archived' | 'pinned' | 'sessions' | string

export type SidebarSessionReleaseDropDecision =
  | { type: 'archive'; sessionId: string }
  | { type: 'none' }
  | { type: 'open'; section: 'pinned' | 'sessions' }
  | { index?: number; pinId: string; type: 'pin' }
  | { nextOrder?: string[]; restoreSessionId?: string; type: 'sessions'; unpinPinId?: string }

interface ResolveSidebarSessionReleaseDropOptions {
  anchor: null | SessionDropAnchor
  anchorPinIdForSessionId: (sessionId: string) => string
  payload: SessionDragPayload
  pinnedSessionIds: readonly string[]
  sectionKey?: null | SidebarSessionReleaseSection
  sessionOrderIds: readonly string[]
  showAllProfiles: boolean
}

function placeSessionIdAtEnd(ids: readonly string[], movingId: string): string[] {
  return [...ids.filter(id => id !== movingId), movingId]
}

export function resolveSidebarSessionReleaseDrop({
  anchor,
  anchorPinIdForSessionId,
  payload,
  pinnedSessionIds,
  sectionKey,
  sessionOrderIds,
  showAllProfiles
}: ResolveSidebarSessionReleaseDropOptions): SidebarSessionReleaseDropDecision {
  if (sectionKey === 'pinned') {
    if (payload.archived) {
      return { type: 'none' }
    }

    const pinId = payload.pinId ?? payload.id
    let index: number | undefined

    if (anchor) {
      const anchorPinId = anchorPinIdForSessionId(anchor.sessionId)

      const nextPinnedIds = placeSessionIdAtAnchor(pinnedSessionIds, pinId, {
        before: anchor.before,
        sessionId: anchorPinId
      })

      if (nextPinnedIds) {
        index = nextPinnedIds.indexOf(pinId)
      } else if (anchorPinId === pinId) {
        return { section: 'pinned', type: 'open' }
      }
    }

    return { index, pinId, type: 'pin' }
  }

  if (sectionKey === 'sessions') {
    const decision: SidebarSessionReleaseDropDecision = { type: 'sessions' }

    if (payload.archived) {
      decision.restoreSessionId = payload.id
    } else if (payload.pinned) {
      decision.unpinPinId = payload.pinId ?? payload.id
    }

    if (!showAllProfiles) {
      const nextOrder = anchor
        ? placeSessionIdAtAnchor(sessionOrderIds, payload.id, anchor)
        : sessionOrderIds.includes(payload.id)
          ? null
          : placeSessionIdAtEnd(sessionOrderIds, payload.id)

      if (nextOrder) {
        decision.nextOrder = nextOrder
      } else if (anchor?.sessionId === payload.id) {
        return { section: 'sessions', type: 'open' }
      }
    }

    return decision
  }

  if (sectionKey === 'archived' && !payload.archived) {
    return { sessionId: payload.id, type: 'archive' }
  }

  return { type: 'none' }
}
