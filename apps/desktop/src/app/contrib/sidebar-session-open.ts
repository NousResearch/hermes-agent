import type { SessionInfo } from '@/hermes'
import { forgetSessionOwnerHintsForSession, requestSessionResume, sessionOwnerRouteFromRow } from '@/store/session'
import { prepareSessionOwnerRetarget } from '@/store/session-states'
import { $sidebarSessionsOpenInNewTab } from '@/store/sidebar-open-preference'
import { canOpenSessionWindow } from '@/store/windows'

import { openSession, type OpenSessionIntent, type OpenSessionNavigate } from '../open-session'

/**
 * Resume through the sidebar row's exact owner when it carries one, then apply
 * the sidebar's navigation policy without weakening the ambient fallback.
 */
export function openSidebarSession(
  sessionId: string,
  session: SessionInfo | undefined,
  navigate: OpenSessionNavigate,
  intent?: Extract<OpenSessionIntent, 'tab' | 'window'>
): void {
  const ownerRoute = sessionOwnerRouteFromRow(session)
  const placement = intent ?? ($sidebarSessionsOpenInNewTab.get() ? 'tab' : 'main')
  const effectivePlacement = placement === 'window' && !canOpenSessionWindow() ? 'tab' : placement

  if (effectivePlacement !== 'window') {
    prepareSessionOwnerRetarget(sessionId, ownerRoute, effectivePlacement === 'main')
  }

  if (ownerRoute) {
    if (effectivePlacement === 'main') {
      requestSessionResume(sessionId, ownerRoute)
    }
  } else {
    forgetSessionOwnerHintsForSession(sessionId)

    if (effectivePlacement === 'main') {
      requestSessionResume(sessionId)
    }
  }

  openSession(
    sessionId,
    navigate,
    effectivePlacement,
    ownerRoute ? { ownerRoute, workspaceMode: 'sessions' } : { workspaceMode: 'sessions' }
  )
}
