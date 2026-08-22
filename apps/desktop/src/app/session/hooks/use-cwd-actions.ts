import { type MutableRefObject, useCallback } from 'react'

import { useI18n } from '@/i18n'
import { notify, notifyError } from '@/store/notifications'
import {
  $currentCwd,
  $newChatWorkspaceTargetGeneration,
  setCurrentBranch,
  setCurrentCwd,
  setNewChatWorkspaceTarget
} from '@/store/session'
import type { SessionRuntimeInfo } from '@/types/hermes'

interface CwdActionsOptions {
  activeSessionIdRef: MutableRefObject<string | null>
  onSessionRuntimeInfo?: (sessionId: string, info: Pick<SessionRuntimeInfo, 'branch' | 'cwd'>) => void
  requestGateway: <T = unknown>(method: string, params?: Record<string, unknown>) => Promise<T>
}

export function useCwdActions({ activeSessionIdRef, onSessionRuntimeInfo, requestGateway }: CwdActionsOptions) {
  const { t } = useI18n()
  const copy = t.desktop

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
    async (cwd: string, sessionIdOverride?: string) => {
      const trimmed = cwd.trim()

      if (!trimmed) {
        return
      }

      // Prefer the caller's target session (the statusbar operates on the
      // FOCUSED session — a tile, or the primary), falling back to the ref so
      // existing primary-scoped callers keep working. The ref alone would
      // re-anchor the primary's workspace even when the statusbar is
      // describing a focused tile (mis-associating session state).
      const sessionId = sessionIdOverride ?? activeSessionIdRef.current

      if (!sessionId) {
        setCurrentCwd(trimmed)
        const workspaceGeneration = setNewChatWorkspaceTarget(trimmed)

        try {
          const info = await requestGateway<{ branch?: string; cwd?: string }>('config.get', {
            key: 'project',
            cwd: trimmed
          })

          if ($newChatWorkspaceTargetGeneration.get() !== workspaceGeneration || activeSessionIdRef.current) {
            return
          }

          // Adopt the backend's normalized cwd so the persisted workspace and
          // branch stay consistent with what the agent will use.
          if (info.cwd) {
            setCurrentCwd(info.cwd)
            setNewChatWorkspaceTarget(info.cwd)
          }

          setCurrentBranch(info.branch || '')
        } catch {
          if ($newChatWorkspaceTargetGeneration.get() === workspaceGeneration && !activeSessionIdRef.current) {
            setCurrentBranch('')
          }
        }

        return
      }

      try {
        const info = await requestGateway<SessionRuntimeInfo>('session.cwd.set', {
          session_id: sessionId,
          cwd: trimmed
        })

        setCurrentCwd(info.cwd || trimmed)
        setCurrentBranch(info.branch || '')
        onSessionRuntimeInfo?.(sessionId, { branch: info.branch || '', cwd: info.cwd || trimmed })
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err)

        if (!message.includes('unknown method')) {
          notifyError(err, copy.cwdChangeFailed)

          return
        }

        setCurrentCwd(trimmed)
        setCurrentBranch('')
        notify({
          kind: 'warning',
          title: copy.cwdStagedTitle,
          message: copy.cwdStagedMessage
        })
      }
    },
    [activeSessionIdRef, copy, onSessionRuntimeInfo, requestGateway]
  )

  return { changeSessionCwd, refreshProjectBranch }
}
