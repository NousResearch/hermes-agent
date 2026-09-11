import { $activeSessionId, $selectedStoredSessionId } from '@/app/chat/session-store'
import { activeGateway } from '@/gateway'
import { notify, notifyError } from '@/app/notifications'
import { renameSession } from '@/api/sessions'

const inFlightRetitles = new Set<string>()

export interface RetitleSessionOptions {
  sessionId?: string | null
  profile?: string
  t?: (key: string) => string
}

export async function runSessionRetitle({
  sessionId,
  profile,
  t,
}: RetitleSessionOptions): Promise<boolean> {
  const targetId = sessionId || $selectedStoredSessionId.get() || $activeSessionId.get()
  if (!targetId) {
    notify({
      kind: 'warning',
      message: t ? t('sessionActions.noActiveSession') : 'No active session selected',
    })
    return false
  }

  if (inFlightRetitles.has(targetId)) {
    return false
  }

  inFlightRetitles.add(targetId)

  const infoMsg = t ? t('sessionActions.regeneratingTitle') : 'Regenerating session title...'
  notify({ kind: 'info', message: infoMsg })

  try {
    const gateway = activeGateway()
    const runtimeId = targetId === $selectedStoredSessionId.get() ? $activeSessionId.get() : targetId

    // Try slash/command dispatch first if autotitler or retitle command exists
    let res: { output?: string } | null = null
    if (gateway?.request) {
      try {
        res = await gateway.request<{ output?: string }>('command.dispatch', {
          name: 'autotitler',
          arg: `rename-now ${targetId}`,
          session_id: runtimeId || targetId,
        })
      } catch {
        // Fall back to server/retitle RPC or empty title clear
        try {
          res = await gateway.request<{ output?: string }>('session.retitle', {
            session_id: runtimeId || targetId,
          })
        } catch {
          // Fall back to REST/RPC rename refresh
          await renameSession(targetId, '', profile)
        }
      }
    } else {
      await renameSession(targetId, '', profile)
    }

    const successMsg = res?.output || (t ? t('sessionActions.regenerateTitleSuccess') : 'Session title regenerated')
    notify({ kind: 'success', message: successMsg })
    return true
  } catch (err) {
    notifyError(err)
    return false
  } finally {
    inFlightRetitles.delete(targetId)
  }
}
