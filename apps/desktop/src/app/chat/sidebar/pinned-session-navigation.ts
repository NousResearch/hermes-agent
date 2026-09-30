import type { SessionInfo } from '@/hermes'

export const PINNED_SESSION_NAV_EVENT = 'hermes:navigate-pinned-session'

export type PinnedSessionDirection = -1 | 1

type PinnedSessionIdentity = Pick<SessionInfo, '_lineage_root_id' | 'id'>

function matchesStoredId(session: PinnedSessionIdentity, storedId: null | string): boolean {
  return storedId != null && (session.id === storedId || session._lineage_root_id === storedId)
}

export function nextPinnedSession<T extends PinnedSessionIdentity>(
  sessions: readonly T[],
  activeStoredId: null | string,
  direction: PinnedSessionDirection
): null | T {
  if (sessions.length === 0) {
    return null
  }

  const activeIndex = sessions.findIndex(session => matchesStoredId(session, activeStoredId))

  if (activeIndex < 0) {
    return direction > 0 ? sessions[0] : sessions[sessions.length - 1]
  }

  return sessions[(activeIndex + direction + sessions.length) % sessions.length]
}

export function requestPinnedSessionNavigation(direction: PinnedSessionDirection): void {
  window.dispatchEvent(new CustomEvent<PinnedSessionDirection>(PINNED_SESSION_NAV_EVENT, { detail: direction }))
}
