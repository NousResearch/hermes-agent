import type { SessionInfo } from '@/hermes'
import { forgetSessionOwnerHintsForSession, requestSessionResume, sessionOwnerRouteFromRow } from '@/store/session'
import { prepareSessionOwnerRetarget } from '@/store/session-states'
import { $sidebarSessionsOpenInNewTab } from '@/store/sidebar-open-preference'

import { openSession, type OpenSessionNavigate } from '../open-session'

/**
 * Resume through the sidebar row's exact owner when it carries one, then apply
 * the sidebar's navigation policy without weakening the ambient fallback.
 */
export function openSidebarSession(
  sessionId: string,
  session: SessionInfo | undefined,
  navigate: OpenSessionNavigate
): void {
  const ownerRoute = sessionOwnerRouteFromRow(session)

  const intent = $sidebarSessionsOpenInNewTab.get() ? 'tab' : 'main'

  if (ownerRoute) {
    prepareSessionOwnerRetarget(sessionId, ownerRoute, intent === 'main')
    requestSessionResume(sessionId, ownerRoute)
  } else {
    forgetSessionOwnerHintsForSession(sessionId)
    requestSessionResume(sessionId)
  }

  openSession(
    sessionId,
    navigate,
    intent,
    ownerRoute ? { ownerRoute, workspaceMode: 'sessions' } : { workspaceMode: 'sessions' }
  )
}
