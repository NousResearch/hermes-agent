import type { SessionInfo } from '@/hermes'
import { normalizeProfileKey } from '@/store/profile'
import { sessionActivityKey } from '@/store/session-activity'

export const sidebarSessionScopeKey = (session: Pick<SessionInfo, 'id' | 'profile'>): string =>
  sessionActivityKey(session.profile, session.id)

export function isSidebarSessionSelected(
  session: Pick<SessionInfo, 'id' | 'profile'>,
  selectedSessionId: null | string,
  selectedProfile: null | string | undefined
): boolean {
  return (
    session.id === selectedSessionId && normalizeProfileKey(session.profile) === normalizeProfileKey(selectedProfile)
  )
}

export function isSidebarSessionWorking(
  session: Pick<SessionInfo, '_lineage_root_id' | 'id' | 'profile'>,
  workingSessionScopeKeys: ReadonlySet<string>
): boolean {
  return (
    workingSessionScopeKeys.has(sidebarSessionScopeKey(session)) ||
    Boolean(
      session._lineage_root_id &&
      workingSessionScopeKeys.has(sessionActivityKey(session.profile, session._lineage_root_id))
    )
  )
}
