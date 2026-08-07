import type { MutableRefObject } from 'react'

import { followActiveSessionCwd, resolveNewSessionCwd } from '@/store/projects'
import {
  $newChatWorkspaceTargetGeneration,
  setCurrentBranch,
  setCurrentCwd,
  setNewChatWorkspaceTarget
} from '@/store/session'

interface WorkspaceSessionOptions {
  activeSessionIdRef: MutableRefObject<string | null>
  followActiveSessionCwd?: (cwd: string) => void | Promise<void>
  onExplicitWorkspace?: (cwd: string) => void
  path: null | string
  requestGateway: <T>(method: string, params?: Record<string, unknown>) => Promise<T>
  startFreshSessionDraft: (options?: { workspaceTarget: string }) => void
}

export function startWorkspaceSession({
  activeSessionIdRef,
  followActiveSessionCwd: followCwd = followActiveSessionCwd,
  onExplicitWorkspace,
  path,
  requestGateway,
  startFreshSessionDraft
}: WorkspaceSessionOptions): void {
  // A worktree lane carries its own path; a project trunk can be path-less, so
  // fall back to the active project's root for that existing controller path.
  const explicitTarget = path?.trim()
  const target = explicitTarget || resolveNewSessionCwd()

  startFreshSessionDraft(target ? { workspaceTarget: target } : undefined)

  if (!target) {
    return
  }

  const workspaceGeneration = $newChatWorkspaceTargetGeneration.get()

  setCurrentCwd(target)
  void requestGateway<{ branch?: string; cwd?: string }>('config.get', { key: 'project', cwd: target })
    .then(info => {
      if ($newChatWorkspaceTargetGeneration.get() !== workspaceGeneration || activeSessionIdRef.current) {
        return
      }

      const resolved = info.cwd || target

      setCurrentCwd(resolved)
      setNewChatWorkspaceTarget(resolved)
      setCurrentBranch(info.branch || '')

      if (explicitTarget) {
        onExplicitWorkspace?.(resolved)
        void followCwd(resolved)
      }
    })
    .catch(() => {
      if ($newChatWorkspaceTargetGeneration.get() === workspaceGeneration && !activeSessionIdRef.current) {
        setCurrentBranch('')
      }
    })
}
