import { useStore } from '@nanostores/react'
import { useEffect, useMemo, useState } from 'react'

import type { HermesGitWorktree } from '@/global'
import type { SessionInfo } from '@/hermes'
import { desktopGit } from '@/lib/desktop-git'
import { mapPool } from '@/lib/pool'
import { $sidebarWorkspaceNodeOpen, toggleWorkspaceNodeCollapsed } from '@/store/layout'
import { $worktreeRefreshToken } from '@/store/projects'

import { sessionRecency, type SidebarProjectTree } from './workspace-groups'

// Page size when revealing more already-loaded rows within a workspace group.
export const SIDEBAR_GROUP_PAGE = 5

// Recent sessions previewed under each project in the overview.
export const PROJECT_PREVIEW_COUNT = 3

// Rows each "Show more" adds once a project's full list is open (an expanded
// overview row, an entered lane, entered Home). Large enough that 50+ sessions
// are a click or two away, small enough that a 5000-chat Home never mounts
// every row at once.
export const PROJECT_SESSION_PAGE = 50

// Reveal `rows` a page at a time: the first `first`, then PROJECT_SESSION_PAGE
// per `showMore()`. `more` is the next step's size (0 once everything shows).
export function useRevealedRows<T>(rows: T[], first: number): { more: number; shown: T[]; showMore: () => void } {
  const [count, setCount] = useState(first)
  const shown = rows.length > count ? rows.slice(0, count) : rows

  return {
    more: Math.min(PROJECT_SESSION_PAGE, rows.length - shown.length),
    shown,
    showMore: () => setCount(current => current + PROJECT_SESSION_PAGE)
  }
}

// Max concurrent `git worktree list` probes when a project spans many repos.
const WORKTREE_PROBE_CONCURRENCY = 4

const pathListKey = (paths: string[]): string =>
  paths
    .map(path => path.trim())
    .filter(Boolean)
    .sort((a, b) => a.localeCompare(b))
    .join('\n')

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

// The overview payload carries only the most-recent preview rows per project.
// Once the user asks for the rest, the project's full lanes are fetched on
// demand; fold them under the (live-overlaid) preview so a just-created
// session keeps its place and nothing renders twice.
export const expandedProjectSessions = (preview: SessionInfo[], hydrated: SidebarProjectTree): SessionInfo[] => {
  const seen = new Set(preview.map(session => session.id))

  return [...preview, ...latestProjectSessions(hydrated, Infinity).filter(session => !seen.has(session.id))]
}

// Home is a fixture, not a project: it always leads the overview, above the
// active project and outside any hand-picked order.
const homeFirst = (projects: SidebarProjectTree[]): SidebarProjectTree[] =>
  projects[0]?.isNoProject || !projects.some(project => project.isNoProject)
    ? projects
    : [...projects.filter(project => project.isNoProject), ...projects.filter(project => !project.isNoProject)]

export function sortProjectsForOverview(
  projects: SidebarProjectTree[],
  activeProjectId: null | string
): SidebarProjectTree[] {
  const sorted = [...projects].sort((a, b) => {
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

    return (
      projectActivityTime(b) - projectActivityTime(a) ||
      a.label.localeCompare(b.label, undefined, { sensitivity: 'base' })
    )
  })

  return homeFirst(sorted)
}

// Layer the user's manual drag-order over the deterministic sort.
//
// This can't just be `orderByIds`: that surfaces every id missing from the saved
// order at the TOP, which is right for sessions (a new chat should not sink) but
// wrong here. The overview also lists repos found by the disk scan that have
// zero Hermes sessions, and those arrive continuously — so once the user dragged
// anything, every freshly-scanned checkout jumped above the projects they
// actually work in.
//
// Fresh projects keep their place in the deterministic sort instead: ones with
// real activity go on top (a project you just started still surfaces), and
// zero-session discoveries sink below the hand-ordered list.
export function orderProjectsByIds(projects: SidebarProjectTree[], orderIds: string[]): SidebarProjectTree[] {
  if (!orderIds.length) {
    return projects
  }

  const byId = new Map(projects.map(project => [project.id, project]))
  const ordered = orderIds.map(id => byId.get(id)).filter((p): p is SidebarProjectTree => Boolean(p))
  const seen = new Set(ordered.map(project => project.id))
  const fresh = projects.filter(project => !seen.has(project.id))

  if (!fresh.length) {
    return homeFirst(ordered)
  }

  return homeFirst([
    ...fresh.filter(project => project.sessionCount > 0),
    ...ordered,
    ...fresh.filter(project => project.sessionCount <= 0)
  ])
}

// Nest each project under its folder-ancestor, keeping the incoming order within every level.
//
// The result stays a FLAT, parent-first list — the overview renders that order and indents the rows
// whose project names a parent (`project.parentId`), so nothing downstream (virtualisation, drag
// order, owner maps) has to learn a nested shape. A `parentId` naming an absent project is treated
// as top level rather than dropping the row, so filtering or dismissing a parent can't orphan a
// child out of the sidebar. Membership is untouched: the child keeps its own sessions.
export function nestProjectsByParent(projects: SidebarProjectTree[]): SidebarProjectTree[] {
  const present = new Set(projects.map(project => project.id))

  const parentOf = (project: SidebarProjectTree): null | string =>
    project.parentId && project.parentId !== project.id && present.has(project.parentId)
      ? project.parentId
      : null

  const children = new Map<string, SidebarProjectTree[]>()

  const nested = projects.filter(project => parentOf(project))

  // Nothing nests: hand the caller the same list it passed in (same contract as
  // `orderProjectsByIds`), so an unnested sidebar keeps referential stability.
  if (!nested.length) {
    return projects
  }

  for (const project of nested) {
    const parent = parentOf(project)

    if (parent) {
      children.set(parent, [...(children.get(parent) ?? []), project])
    }
  }

  const out: SidebarProjectTree[] = []
  const placed = new Set<string>()

  const push = (project: SidebarProjectTree): void => {
    if (placed.has(project.id)) {
      return
    }

    placed.add(project.id)
    out.push(project)

    for (const child of children.get(project.id) ?? []) {
      push(child)
    }
  }

  for (const project of projects) {
    if (!parentOf(project)) {
      push(project)
    }
  }

  // Belt and braces: a cycle in `parentId` would leave every member rootless and silently drop rows.
  for (const project of projects) {
    push(project)
  }

  return out
}

/**
 * The dragged project plus everything nested under it, transitively, and the project itself — so
 * `has(id)` answers "is this that project, or one of its descendants?".
 */
export function projectDescendantIds(
  projects: Pick<SidebarProjectTree, 'id' | 'parentId'>[],
  id: string
): Set<string> {
  const children = new Map<string, string[]>()

  for (const project of projects) {
    if (project.parentId) {
      children.set(project.parentId, [...(children.get(project.parentId) ?? []), project.id])
    }
  }

  const seen = new Set<string>([id])
  const queue = [...(children.get(id) ?? [])]

  while (queue.length) {
    const next = queue.pop() as string

    if (seen.has(next)) {
      continue
    }

    seen.add(next)
    queue.push(...(children.get(next) ?? []))
  }

  return seen
}

/**
 * Where the entered view's back row goes: one level up — the parent of the project you are inside —
 * or `null` for the overview when it is a top-level one. The mirror of entering a nested row, so a
 * drill-down walks back out a step at a time.
 */
export function projectBackTarget(
  projects: Pick<SidebarProjectTree, 'id' | 'parentId'>[],
  id: string
): null | string {
  const parentId = projects.find(project => project.id === id)?.parentId

  // A parent id that names no present project is no level to step back into (the same tolerance
  // `nestProjectsByParent` has for a filtered-out parent): the overview is the answer.
  return parentId && projects.some(project => project.id === parentId) ? parentId : null
}

/**
 * Every project's nesting depth along its `parentId` chain — 0 for a top-level row, 1 for its child,
 * 2 for its grandchild. The overview indents a row by this, so a subproject of a subproject is drawn
 * one step further in than the subproject itself rather than beside it. Cycle-safe, and a `parentId`
 * naming an absent project counts as top level — the same tolerance `visibleProjectRows` and
 * `projectBackTarget` have.
 */
export function projectDepths(projects: Pick<SidebarProjectTree, 'id' | 'parentId'>[]): Map<string, number> {
  const byId = new Map(projects.map(project => [project.id, project]))
  const depths = new Map<string, number>()

  for (const project of projects) {
    const seen = new Set<string>([project.id])
    let depth = 0
    let parentId = project.parentId

    while (parentId && !seen.has(parentId)) {
      seen.add(parentId)

      const parent = byId.get(parentId)

      if (!parent) {
        break
      }

      depth += 1
      parentId = parent.parentId
    }

    depths.set(project.id, depth)
  }

  return depths
}

/**
 * Every session a project stands for: its own rows plus the rows of every project nested under it,
 * transitively. This is what a collapsed row folds its status up from — ownership is deepest-wins, so
 * a subproject's sessions belong to the subproject and are absent from its ancestors' `sessionIds`.
 */
export function projectSubtreeSessionIds(projects: SidebarProjectTree[], id: string): string[] {
  const subtree = projectDescendantIds(projects, id)
  const ids: string[] = []

  for (const project of projects) {
    if (subtree.has(project.id)) {
      ids.push(...(project.sessionIds ?? []))
    }
  }

  return ids
}

/**
 * The projects whose row should be on screen: a project whose ancestor chain is all open.
 *
 * Nesting is a display grouping, so a collapsed container hides everything under it — a subproject
 * goes away with its parent, and comes back in whatever state it was left in (each row keeps its own
 * open flag, keyed by project id, so expanding a child and collapsing its parent is not a reset).
 * A parent that is not in this list (archived, or a stale id) cannot hide anything.
 */
export function visibleProjectRows(
  projects: SidebarProjectTree[],
  isOpen: (id: string) => boolean
): SidebarProjectTree[] {
  const byId = new Map(projects.map(project => [project.id, project]))

  return projects.filter(project => {
    const seen = new Set<string>()
    let parentId = project.parentId

    while (parentId && !seen.has(parentId)) {
      seen.add(parentId)

      const parent = byId.get(parentId)

      if (!parent) {
        break
      }

      if (!isOpen(parent.id)) {
        return false
      }

      parentId = parent.parentId
    }

    return true
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
    const git = desktopGit()

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
// State is stored as the RESOLVED boolean per node (see `$sidebarWorkspaceNodeOpen`),
// so a node whose `defaultOpen` flips — an empty worktree/branch lane defaults
// collapsed, then defaults open once it holds a session — keeps whatever the
// user explicitly chose instead of having it silently reinterpreted. An absent
// id follows `defaultOpen`, so empty lanes still start collapsed until opened.
export function useWorkspaceNodeOpen(id: string, defaultOpen = true): [boolean, () => void] {
  const state = useStore($sidebarWorkspaceNodeOpen)

  return [state[id] ?? defaultOpen, () => toggleWorkspaceNodeCollapsed(id, defaultOpen)]
}
