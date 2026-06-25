import { useStore } from '@nanostores/react'
import { useEffect, useMemo, useState } from 'react'

import type { HermesGitWorktree } from '@/global'
import type { SessionInfo } from '@/hermes'
import { mapPool } from '@/lib/pool'
import { $sidebarWorkspaceCollapsedIds, toggleWorkspaceNodeCollapsed } from '@/store/layout'
import { $worktreeRefreshToken } from '@/store/projects'

import { sessionRecency, type SidebarProjectTree } from './workspace-groups'

// Page size when revealing more already-loaded rows within a workspace group.
export const SIDEBAR_GROUP_PAGE = 5

// Recent sessions previewed under each project in the overview.
export const PROJECT_PREVIEW_COUNT = 3

// Max concurrent `git worktree list` probes when a project spans many repos.
const WORKTREE_PROBE_CONCURRENCY = 4

const pathListKey = (paths: string[]): string =>
  paths.map(path => path.trim()).filter(Boolean).sort((a, b) => a.localeCompare(b)).join('\n')

// Every session in a project, across its repos/worktrees (order-agnostic).
const projectSessions = (project: SidebarProjectTree): SessionInfo[] =>
  project.repos.flatMap(repo => repo.groups.flatMap(group => group.sessions))

export const projectTreeCwd = (project: SidebarProjectTree): null | string =>
  project.path || project.repos.find(repo => repo.path)?.path || null

// Overview rows carry their activity stamp from the backend (lanes are empty in
// overview mode), falling back to loaded session times when present.
const projectActivityTime = (project: SidebarProjectTree): number =>
  Math.max(
    project.lastActive ?? 0,
    projectSessions(project).reduce((latest, s) => Math.max(latest, sessionRecency(s)), 0)
  )

// The project's most-recent sessions, for the overview preview under each row.
export const latestProjectSessions = (project: SidebarProjectTree, limit: number): SessionInfo[] =>
  [...projectSessions(project)].sort((a, b) => sessionRecency(b) - sessionRecency(a)).slice(0, limit)

export function sortProjectsForOverview(
  projects: SidebarProjectTree[],
  activeProjectId: null | string
): SidebarProjectTree[] {
  return [...projects].sort((a, b) => {
    const aActive = Boolean(activeProjectId && a.id === activeProjectId && !a.isAuto)
    const bActive = Boolean(activeProjectId && b.id === activeProjectId && !b.isAuto)

    if (aActive !== bActive) {
      return aActive ? -1 : 1
    }

    if (!a.isAuto !== !b.isAuto) {
      return a.isAuto ? 1 : -1
    }

    const aHasSessions = a.sessionCount > 0
    const bHasSessions = b.sessionCount > 0

    if (aHasSessions !== bHasSessions) {
      return aHasSessions ? -1 : 1
    }

    return projectActivityTime(b) - projectActivityTime(a) || a.label.localeCompare(b.label, undefined, { sensitivity: 'base' })
  })
}

// Project drill-in lanes are git-driven: source them from `git worktree list` so
// linked worktrees still appear even when their sessions aren't in the recents
// payload currently loaded in memory.
export function useRepoWorktreeMap(
  repoPaths: string[],
  enabled: boolean
): [Record<string, HermesGitWorktree[]>, boolean] {
  const [map, setMap] = useState<Record<string, HermesGitWorktree[]>>({})
  const [loading, setLoading] = useState(false)
  const key = useMemo(() => pathListKey(repoPaths), [repoPaths])
  // Refetch when a worktree is added/removed so a new lane shows immediately.
  const refreshToken = useStore($worktreeRefreshToken)

  useEffect(() => {
    const git = window.hermesDesktop?.git

    if (!enabled || !repoPaths.length || !git?.worktreeList) {
      setMap({})
      setLoading(false)

      return
    }

    let cancelled = false

    setLoading(true)
    // Bounded so a many-repo project doesn't spawn a `git` process per repo at once.
    void mapPool(repoPaths, WORKTREE_PROBE_CONCURRENCY, async repoPath => {
      try {
        return [repoPath, await git.worktreeList(repoPath)] as const
      } catch {
        return [repoPath, []] as const
      }
    })
      .then(entries => void (cancelled || setMap(Object.fromEntries(entries))))
      .finally(() => void (cancelled || setLoading(false)))

    return () => {
      cancelled = true
    }
  }, [enabled, key, repoPaths, refreshToken])

  return [map, loading]
}

// Persisted open/collapse for a repo/worktree node. Lets a project's folder
// layout auto-restore when you enter it, and survive reloads.
//
// The persisted set is an OVERRIDE of `defaultOpen`, not an absolute "collapsed"
// list: XOR lets one store serve both polarities. A default-open node (repo,
// populated lane) lists collapses; a default-collapsed node (an EMPTY lane — no
// sessions yet) instead records an explicit expand. So empty worktree/branch
// lanes start collapsed and only open when the user clicks in.
export function useWorkspaceNodeOpen(id: string, defaultOpen = true): [boolean, () => void] {
  const collapsed = useStore($sidebarWorkspaceCollapsedIds)
  const overridden = collapsed.includes(id)

  return [defaultOpen ? !overridden : overridden, () => toggleWorkspaceNodeCollapsed(id)]
}
