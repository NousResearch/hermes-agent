import type { SessionTitleResponse } from '@/app/types'
import { translateNow } from '@/i18n'
import { activeGateway } from '@/store/gateway'
import { notify, notifyError } from '@/store/notifications'
import {
  $activeSessionId,
  $selectedStoredSessionId,
  sessionMatchesStoredId,
  setSessions
} from '@/store/session'

export interface RetitleSessionOptions {
  sessionId?: string
  profile?: string
}

/** Regenerate the currently selected session title through the canonical backend RPC. */
export async function runSessionRetitle(options: RetitleSessionOptions = {}): Promise<null | string> {
  const t = translateNow()
  const storedSessionId = $selectedStoredSessionId.get()
  const runtimeSessionId = $activeSessionId.get()

  // The Desktop surface intentionally supports only the active session. The
  // menu is disabled for every other row; this re-check covers a selection
  // change between opening the menu and clicking the action.
  if (!storedSessionId || !runtimeSessionId || (options.sessionId && options.sessionId !== storedSessionId)) {
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
