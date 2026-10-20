import type { MutableRefObject } from 'react'

import { pinNewChatProfile } from '@/store/profile'
import { followActiveSessionCwd, projectProfile, resolveNewSessionCwd } from '@/store/projects'
import {
  $newChatWorkspaceTargetGeneration,
  type NewChatWorkspaceTarget,
  setCurrentBranch,
  setCurrentCwd,
  setNewChatWorkspaceTarget
} from '@/store/session'

/**
 * Pin the next session create to the profile the project tree is rendered
 * under. Every "+"-on-a-project entry point must run this before it branches
 * into its own create path — the occupied-chat tile path skips
 * `startWorkspaceSession` entirely, so the pin cannot live only inside it
 * (#124265).
 */
export function pinNewSessionWorkspaceProfile(): null | string {
  // The project tree is rendered under one profile; the "+" belongs to it.
  // Pin that intent now — otherwise desktopSessionCreateParams falls back to
  // $activeGatewayProfile, which a still-settling profile swap can move
  // between this click and Send (#79005). All-profiles view has no owner.
  const profile = projectProfile()

  if (profile) {
    pinNewChatProfile(profile)
  }

  return profile
}

interface SessionInWorkspaceDeps {
  activeSessionIdRef: MutableRefObject<string | null>
  /** Pre-computed `mainChatOccupied()` — whether a loaded chat must be kept. */
  chatOccupied: boolean
  followActiveSessionCwd?: (cwd: string) => void | Promise<void>
  onExplicitWorkspace?: (cwd: string) => void
  openNewSessionTile: (options: { cwd: null | string; listed: boolean }) => void | Promise<void>
  requestGateway: <T>(method: string, params?: Record<string, unknown>) => Promise<T>
  setWorkspaceScope: (scope: 'sessions') => void
  startFreshSessionDraft: (options?: { workspaceTarget: NewChatWorkspaceTarget }) => void
}

/**
 * The sidebar/project "+" door. Both create paths must carry the pinning
 * profile: once a chat is loaded the "+" stacks a tile (which resolves its
 * owner from mutable new-chat state) instead of going through the fresh-draft
 * flow, so the pin has to fire before that branch (#124265).
 */
export function startSessionInWorkspace(
  { activeSessionIdRef, chatOccupied, followActiveSessionCwd: followCwd, onExplicitWorkspace, openNewSessionTile, requestGateway, setWorkspaceScope, startFreshSessionDraft }: SessionInWorkspaceDeps,
  path: null | string,
  options?: { openTab?: boolean }
): void {
  setWorkspaceScope('sessions')

  const profile = pinNewSessionWorkspaceProfile()

  if (options?.openTab && chatOccupied) {
    void openNewSessionTile({ cwd: path, listed: false })

    return
  }

  startWorkspaceSession({
    activeSessionIdRef,
    followActiveSessionCwd: followCwd,
    onExplicitWorkspace,
    path,
    profile,
    requestGateway,
    startFreshSessionDraft
  })
}

interface WorkspaceSessionOptions {
  activeSessionIdRef: MutableRefObject<string | null>
  followActiveSessionCwd?: (cwd: string) => void | Promise<void>
  onExplicitWorkspace?: (cwd: string) => void
  /** Pre-pinned owner profile from `pinNewSessionWorkspaceProfile()`. */
  profile?: null | string
  path: null | string
  requestGateway: <T>(method: string, params?: Record<string, unknown>) => Promise<T>
  startFreshSessionDraft: (options?: { workspaceTarget: NewChatWorkspaceTarget }) => void
}

export function startWorkspaceSession({
  activeSessionIdRef,
  followActiveSessionCwd: followCwd = followActiveSessionCwd,
  onExplicitWorkspace,
  profile = pinNewSessionWorkspaceProfile(),
  path,
  requestGateway,
  startFreshSessionDraft
}: WorkspaceSessionOptions): void {

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
