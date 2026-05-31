import { type MutableRefObject, useCallback } from 'react'

import { notify, notifyError } from '@/store/notifications'
import { $currentCwd, setCurrentBranch, setCurrentCwd } from '@/store/session'
import type { SessionRuntimeInfo } from '@/types/hermes'

interface CwdActionsOptions {
  activeSessionId: string | null
  activeSessionIdRef: MutableRefObject<string | null>
  onSessionRuntimeInfo?: (info: Pick<SessionRuntimeInfo, 'branch' | 'cwd'>) => void
  requestGateway: <T = unknown>(method: string, params?: Record<string, unknown>) => Promise<T>
}

export function useCwdActions({
  activeSessionId,
  activeSessionIdRef,
  onSessionRuntimeInfo,
  requestGateway
}: CwdActionsOptions) {
  const refreshProjectBranch = useCallback(
    async (cwd: string) => {
      const target = cwd.trim()

      if (!target || activeSessionIdRef.current) {
        return
      }

      try {
        const info = await requestGateway<{ branch?: string; cwd?: string }>('config.get', {
          key: 'project',
          cwd: target
        })

        if (!activeSessionIdRef.current && ($currentCwd.get() || target) === (info.cwd || target)) {
          setCurrentBranch(info.branch || '')
        }
      } catch {
        setCurrentBranch('')
      }
    },
    [activeSessionIdRef, requestGateway]
  )

  const changeSessionCwd = useCallback(
    async (cwd: string) => {
      const trimmed = cwd.trim()

      if (!trimmed) {
        return
      }

      if (!activeSessionId) {
        try {
          const info = await requestGateway<{ branch?: string; cwd?: string }>('config.get', {
            key: 'project',
            cwd: trimmed
          })

          setCurrentCwd(info.cwd || trimmed)
          setCurrentBranch(info.branch || '')
        } catch (err) {
          notifyError(err, 'Working directory change failed')
        }

        return
      }

      try {
        const info = await requestGateway<SessionRuntimeInfo>('session.cwd.set', {
          session_id: activeSessionId,
          cwd: trimmed
        })

        setCurrentCwd(info.cwd || trimmed)
        setCurrentBranch(info.branch || '')
        onSessionRuntimeInfo?.({ branch: info.branch || '', cwd: info.cwd || trimmed })
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err)

        if (!message.includes('unknown method')) {
          notifyError(err, 'Working directory change failed')

          return
        }

        setCurrentCwd(trimmed)
        setCurrentBranch('')
        notify({
          kind: 'warning',
          title: 'Working directory staged',
          message: 'Restart the desktop backend to apply cwd changes to this active session.'
        })
      }
    },
    [activeSessionId, onSessionRuntimeInfo, requestGateway]
  )

  return { changeSessionCwd, refreshProjectBranch }
}
