import type { SessionTitleResponse } from '@/app/types'
import { translateNow } from '@/i18n'
import { activeGateway } from '@/store/gateway'
import { notify, notifyError } from '@/store/notifications'
import {
  $activeSessionId,
  $selectedStoredSessionId,
  $sessions,
  sessionMatchesStoredId,
  setSessions
} from '@/store/session'

export interface RetitleSessionOptions {
  sessionId?: string
}

/** True when a row id and the current selection resolve to the same stored conversation. */
export function sessionRetitleMatchesSelection(sessionId?: string): boolean {
  const selectedStoredSessionId = $selectedStoredSessionId.get()
  if (!sessionId || !selectedStoredSessionId) {
    return false
  }

  const row = $sessions.get().find(session => sessionMatchesStoredId(session, sessionId))
  return Boolean(row && sessionMatchesStoredId(row, selectedStoredSessionId))
}

/** Regenerate the currently selected session title through the canonical backend RPC. */
export async function runSessionRetitle(options: RetitleSessionOptions = {}): Promise<null | string> {
  const t = translateNow()
  const storedSessionId = $selectedStoredSessionId.get()
  const runtimeSessionId = $activeSessionId.get()

  // Snapshot the target before the request. The lineage-aware re-check covers
  // both a menu opened on a stale row and auto-compression id rotation.
  if (!storedSessionId || !runtimeSessionId || (options.sessionId && !sessionRetitleMatchesSelection(options.sessionId))) {
    return null
  }

  const gateway = activeGateway()
  if (!gateway) {
    notifyError(new Error('session.retitle unavailable'), t.sidebar.row.regenerateTitleFailed)
    return null
  }

  notify({ durationMs: 2_000, kind: 'info', message: t.sidebar.row.regeneratingTitle })

  try {
    const result = await gateway.request<SessionTitleResponse>('session.retitle', {
      session_id: runtimeSessionId
    })
    const title = result?.title?.trim()
    if (!title) {
      throw new Error('session.retitle returned no title')
    }

    setSessions(current => {
      let changed = false
      const next = current.map(session => {
        if (!sessionMatchesStoredId(session, storedSessionId) || session.title === title) {
          return session
        }
        changed = true
        return { ...session, title }
      })
      return changed ? next : current
    })

    notify({ durationMs: 2_000, kind: 'success', message: t.sidebar.row.regenerateTitleSuccess })
    return title
  } catch (error) {
    notifyError(error, t.sidebar.row.regenerateTitleFailed)
    return null
  }
}
