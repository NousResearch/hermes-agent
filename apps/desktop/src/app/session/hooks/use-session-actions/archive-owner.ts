import type { ProfileScope } from '@/hermes'
import { sessionMutationScopeFromRow, type SessionOwnerScope } from '@/store/session-request-router'

interface ArchivableSession {
  _desktop_route_binding?: string
  connection_id?: null | string
  id?: null | string
  profile?: null | string
}

type SetArchived = (sessionId: string, archived: boolean, owner: ProfileScope) => Promise<unknown>

export function archiveSessionOwner(
  session: ArchivableSession | null | undefined,
  fallbackProfile: SessionOwnerScope
): ProfileScope {
  return sessionMutationScopeFromRow(session) ?? fallbackProfile
}

export async function archiveSessionWithOwner(
  sessionId: string,
  session: ArchivableSession | null | undefined,
  fallbackProfile: SessionOwnerScope,
  setArchived: SetArchived
) {
  await setArchived(sessionId, true, archiveSessionOwner(session, fallbackProfile))
}
