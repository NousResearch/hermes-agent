import { probeStoredSession } from '@/app/session/hooks/use-session-actions/utils'
import { announceNewSessionDraftKey } from '@/store/composer'
import { $tileViewScope } from '@/store/profile'
import {
  $selectedStoredSessionId,
  forgetSessionOwnerHintsForSession,
  resolveComposerSessionKey,
  sessionMatchesStoredId,
  sessionOwnerRouteFromRow,
  setSessionOwnerHint
} from '@/store/session'
import { forgetViewMain, viewMainSession } from '@/store/session-tile-views'
import type { SessionInfo } from '@/types/hermes'

import { sessionRoute } from '../../routes'

/**
 * Cold-start restore of the FIRST tab (the workspace pane) from the same
 * per-view memory the other tabs use: keyed by connection id and profile view
 * ("All profiles" or one profile), not by the gateway profile and connection
 * URL. A chat outside the sidebar's recent page, or owned by another profile
 * inside "All profiles", reopens like any other tab instead of falling back to
 * a new session.
 */

type RestorableRow = Pick<
  SessionInfo,
  '_lineage_root_id' | 'connection_id' | 'id' | 'parent_session_id' | 'profile' | 'source'
>

type Navigate = (path: string, options: { replace: boolean }) => void

export type MainTabProbe = { id: string; session: SessionInfo; status: 'found' } | { status: 'gone' | 'inconclusive' }

/** One `[main-tab]` line in desktop.log per restore decision (ids only). */
export function logMainTab(event: string, detail: Record<string, null | string> = {}): void {
  window.hermesDesktop?.logLine?.(`[main-tab] ${event} ${JSON.stringify(detail)}`)
}

/** Resolve a remembered first-tab chat that the sidebar page does not list:
 *  the by-id ladder walks every profile, and a delegate child repairs to its
 *  parent (a child is never a user-facing destination, #56983). */
export async function resolveMainTabSession(
  id: string,
  probe: typeof probeStoredSession = probeStoredSession
): Promise<MainTabProbe> {
  const result = await probe(id)

  if (result.status !== 'found') {
    return result
  }

  if (result.session.source === 'subagent') {
    const parent = result.session.parent_session_id

    return parent ? { id: parent, session: result.session, status: 'found' } : { status: 'gone' }
  }

  return { id, session: result.session, status: 'found' }
}

function repairOwnerHint(id: string, row: Pick<SessionInfo, 'connection_id' | 'profile'>): void {
  const ownerRoute = sessionOwnerRouteFromRow(row)

  if (ownerRoute) {
    setSessionOwnerHint(id, ownerRoute)
  } else {
    forgetSessionOwnerHintsForSession(id)
  }
}

/**
 * The cold-start decision ahead of the per-profile restore. True when it is
 * settled here (or must wait); `latch` closes the restore once settled.
 *
 * display.resume_last_session (#60812): hold until the config record answers,
 * then either restore or stay on the fresh chat. Remembered ids keep being
 * written either way, so flipping the switch back on resumes from the very
 * next launch.
 */
export function decideMainTabColdStart(
  resumeLastSession: boolean | undefined,
  sessions: readonly RestorableRow[],
  navigate: Navigate,
  latch: () => void
): boolean {
  if (resumeLastSession === undefined) {
    return true
  }

  const decision = resumeLastSession ? restoreMainTabFromView(sessions, navigate) : 'done'

  if (decision === 'done') {
    latch()
  }

  return decision !== 'unknown'
}

/**
 * Decide the first tab's chat from the view memory. Returns:
 * - `'unknown'`: this view never recorded a first-tab chat (caller keeps the
 *   legacy per-profile restore);
 * - `'hold'`: wait for the session list before deciding;
 * - `'done'`: handled (restored, kept on a new session, or resolving by id).
 */
export function restoreMainTabFromView(
  sessions: readonly RestorableRow[],
  navigate: Navigate
): 'done' | 'hold' | 'unknown' {
  const view = $tileViewScope.get()
  const remembered = viewMainSession(view)

  if (remembered === undefined) {
    return 'unknown'
  }

  if (remembered === null) {
    logMainTab('restore', { decision: 'new-session', view })

    return 'done'
  }

  if (sessions.length === 0) {
    return 'hold'
  }

  const row = sessions.find(session => sessionMatchesStoredId(session, remembered))

  if (row && row.source !== 'subagent') {
    repairOwnerHint(remembered, row)
    announceNewSessionDraftKey(resolveComposerSessionKey(remembered, sessions))
    navigate(sessionRoute(remembered), { replace: true })
    logMainTab('restore', { decision: 'listed', id: remembered, view })

    return 'done'
  }

  void resolveMainTabSession(remembered)
    .then(result => {
      if (result.status === 'gone') {
        forgetViewMain(view, remembered)
        logMainTab('restore', { decision: 'gone', id: remembered, view })

        return
      }

      if (result.status !== 'found') {
        // Backend unreachable or still booting: keep the memory for the next launch.
        logMainTab('restore', { decision: 'inconclusive', id: remembered, view })

        return
      }

      // The person opened a chat while the lookup ran: theirs wins.
      if ($selectedStoredSessionId.get()) {
        logMainTab('restore', { decision: 'superseded', id: remembered, view })

        return
      }

      repairOwnerHint(result.id, result.session)
      announceNewSessionDraftKey(resolveComposerSessionKey(result.id, sessions))
      navigate(sessionRoute(result.id), { replace: true })
      logMainTab('restore', { decision: 'by-id', id: result.id, view })
    })
    .catch(() => logMainTab('restore', { decision: 'error', id: remembered, view }))

  return 'done'
}
