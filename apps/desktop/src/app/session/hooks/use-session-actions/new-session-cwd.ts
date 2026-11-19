import { NO_PROJECT_ID } from '@/app/chat/sidebar/projects/workspace-groups'
import { $projectScope } from '@/store/project-scope'
import { resolveNewSessionCwd } from '@/store/projects'
import { $currentCwd, $newChatWorkspaceTarget, type NewChatWorkspaceTarget } from '@/store/session'

/** The cwd for a new session. An explicit one-shot workspace target (null → detached,
 *  string → that folder) wins; otherwise the live cwd, then the project-aware default
 *  (resolveNewSessionCwd — a project's new session keeps its repo cwd). Home is an
 *  explicit detached scope: do not let a stale live cwd from the previously selected
 *  project leak into this new session (#84220). */
export function newSessionCwd(): string {
  const workspaceTarget = $newChatWorkspaceTarget.get()
  const homeScope = $projectScope.get() === NO_PROJECT_ID

  return workspaceTarget === null || (workspaceTarget === undefined && homeScope)
    ? ''
    : typeof workspaceTarget === 'string'
      ? workspaceTarget.trim()
      : $currentCwd.get().trim() || resolveNewSessionCwd()
}

export function normalizeNewChatWorkspaceTarget(target: NewChatWorkspaceTarget): NewChatWorkspaceTarget {
  return typeof target === 'string' ? target.trim() || null : target
}
