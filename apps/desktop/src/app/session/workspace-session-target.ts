import type { MutableRefObject } from 'react'

import { requestGatewayForAgent } from '@/store/gateway'
import { type AgentProfileRoute, pinNewChatProfile, pinNewChatRoute } from '@/store/profile'
import { followActiveSessionCwd, projectProfile, resolveNewSessionCwd } from '@/store/projects'
import {
  $newChatWorkspaceTargetGeneration,
  type NewChatWorkspaceTarget,
  setCurrentBranch,
  setCurrentCwd,
  setNewChatWorkspaceTarget
} from '@/store/session'

interface WorkspaceSessionOptions {
  activeSessionIdRef: MutableRefObject<string | null>
  followActiveSessionCwd?: (cwd: string) => void | Promise<void>
  onExplicitWorkspace?: (cwd: string) => void
  /** A project another gateway owns: the draft is created there and its folder resolved there. */
  owner?: AgentProfileRoute
  path: null | string
  requestGateway: <T>(method: string, params?: Record<string, unknown>) => Promise<T>
  startFreshSessionDraft: (options?: { workspaceTarget: NewChatWorkspaceTarget }) => void
}

export function startWorkspaceSession({
  activeSessionIdRef,
  followActiveSessionCwd: followCwd = followActiveSessionCwd,
  onExplicitWorkspace,
  owner,
  path,
  requestGateway,
  startFreshSessionDraft
}: WorkspaceSessionOptions): void {
  if (owner) {
    startOwnedWorkspaceSession(owner, path, activeSessionIdRef, startFreshSessionDraft)

    return
  }

  // The project tree is rendered under one profile; the "+" belongs to it.
  // Pin that intent now — otherwise desktopSessionCreateParams falls back to
  // $activeGatewayProfile, which a still-settling profile swap can move
  // between this click and Send (#79005). All-profiles view has no owner.
  const profile = projectProfile()

  if (profile) {
    pinNewChatProfile(profile)
  }

  // Home's "+" passes path=null on purpose ("no folder"). That must stay
  // detached — do NOT fall through to resolveNewSessionCwd(), which can still
  // return a default/remembered project folder and re-attach the last repo
  // (digitwo: New session in Home still shows `main`).
  if (path === null) {
    startFreshSessionDraft({ workspaceTarget: null })

    return
  }

  // A worktree lane carries its own path. Empty string (legacy/path-less trunk)
  // can fall back to the active project's root, but null was handled above.
  const explicitTarget = path.trim()
  const target = explicitTarget || resolveNewSessionCwd()

  startFreshSessionDraft(target ? { workspaceTarget: target } : undefined)

  if (!target) {
    return
  }

  const workspaceGeneration = $newChatWorkspaceTargetGeneration.get()

  setCurrentCwd(target)
  void requestGateway<{ branch?: string; cwd?: string }>('config.get', {
    key: 'project',
    cwd: target,
    // The project's profile decides its terminal backend: an ssh project dir is not on this host, and
    // resolving it under the launch profile would normalize it away to the launch cwd.
    ...(profile ? { profile } : {})
  })
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

/** The folder is on the owner's machine: resolve it there, and leave this window's project scope
 *  alone (the same path here can be a different project). */
function startOwnedWorkspaceSession(
  owner: AgentProfileRoute,
  path: null | string,
  activeSessionIdRef: MutableRefObject<string | null>,
  startFreshSessionDraft: WorkspaceSessionOptions['startFreshSessionDraft']
): void {
  const route = pinNewChatRoute(owner)
  const target = path?.trim() || null

  startFreshSessionDraft({ workspaceTarget: target })

  if (!target) {
    return
  }

  const workspaceGeneration = $newChatWorkspaceTargetGeneration.get()

  setCurrentCwd(target)
  void requestGatewayForAgent<{ branch?: string; cwd?: string }>(route.connectionId, route.profile, 'config.get', {
    key: 'project',
    cwd: target,
    profile: route.profile
  })
    .then(info => {
      if ($newChatWorkspaceTargetGeneration.get() !== workspaceGeneration || activeSessionIdRef.current) {
        return
      }

      setCurrentCwd(info.cwd || target)
      setNewChatWorkspaceTarget(info.cwd || target)
      setCurrentBranch(info.branch || '')
    })
    .catch(() => {
      if ($newChatWorkspaceTargetGeneration.get() === workspaceGeneration && !activeSessionIdRef.current) {
        setCurrentBranch('')
      }
    })
}
