import { type MutableRefObject, useCallback, useRef } from 'react'

import { useI18n } from '@/i18n'
import { notify, notifyError } from '@/store/notifications'
import { ensureProjectForFolder, pickProjectFolder } from '@/store/projects'
import {
  $currentCwd,
  $newChatWorkspaceTargetGeneration,
  setCurrentBranch,
  setCurrentCwd,
  setNewChatWorkspaceTarget
} from '@/store/session'
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
  const { t } = useI18n()
  const choosingProjectFolderRef = useRef(false)
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

  const stageNewChatWorkspace = useCallback(
    async (cwd: string) => {
      if (activeSessionIdRef.current) {
        return
      }

      const workspaceGeneration = setNewChatWorkspaceTarget(cwd)
      setCurrentCwd(cwd)

      try {
        const info = await requestGateway<{ branch?: string; cwd?: string }>('config.get', {
          key: 'project',
          cwd
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
    },
    [activeSessionIdRef, requestGateway]
  )

  const startProjectFromFolder = useCallback(
    async (cwd: string) => {
      if (activeSessionIdRef.current) {
        return
      }

      try {
        // Project ownership is authoritative: do not expose a raw cwd when the
        // existing owner lookup or explicit project creation fails.
        const projectCwd = await ensureProjectForFolder(cwd, () => Boolean(activeSessionIdRef.current))

        // A session could start while the folder picker/create RPC was open.
        // Do not let this detached-chat action mutate its workspace.
        if (projectCwd === null || activeSessionIdRef.current) {
          return
        }

        await stageNewChatWorkspace(projectCwd)
      } catch (error) {
        notifyError(error, copy.cwdChangeFailed)
      }
    },
    [activeSessionIdRef, copy, stageNewChatWorkspace]
  )

  const chooseProjectFolder = useCallback(async () => {
    if (activeSessionIdRef.current || choosingProjectFolderRef.current) {
      return
    }

    choosingProjectFolderRef.current = true

    try {
      const folder = await pickProjectFolder()

      if (folder) {
        await startProjectFromFolder(folder)
      }
    } catch (error) {
      notifyError(error, copy.cwdChangeFailed)
    } finally {
      choosingProjectFolderRef.current = false
    }
  }, [activeSessionIdRef, copy, startProjectFromFolder])

  const changeSessionCwd = useCallback(
    async (cwd: string) => {
      const trimmed = cwd.trim()

      if (!trimmed) {
        return
      }

      if (!activeSessionId) {
        await stageNewChatWorkspace(trimmed)

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
    [activeSessionId, copy, onSessionRuntimeInfo, requestGateway, stageNewChatWorkspace]
  )

  return { changeSessionCwd, chooseProjectFolder, refreshProjectBranch, startProjectFromFolder }
}
