import { useStore } from '@nanostores/react'
import { type RefObject, useCallback, useEffect, useMemo, useRef, useState } from 'react'

import type { HermesGitWorktree } from '@/global'
import type { SessionInfo } from '@/hermes'
import { useResizeObserver } from '@/hooks/use-resize-observer'
import { desktopGit } from '@/lib/desktop-git'
import { mapPool } from '@/lib/pool'
import { $sidebarWorkspaceNodeOpen, toggleWorkspaceNodeCollapsed } from '@/store/layout'
import { $worktreeRefreshToken } from '@/store/projects'
import type { SessionListDensity } from '@/store/session-list-density'

import { sessionRowEstimate } from '../session-row-details'

import { sessionRecency, type SidebarProjectTree } from './workspace-groups'

// Page size when revealing more already-loaded rows within a workspace group.
export const SIDEBAR_GROUP_PAGE = 5

// Recent sessions VISIBLE under each project in the overview. The preview is a
// glance, not a list — three rows is the height it gets. How many it HOLDS is
// `PROJECT_PREVIEW_LOADED` (workspace-groups.ts), shared with the store so the
// fetch depth and the render depth cannot drift.
export const PROJECT_PREVIEW_COUNT = 3

// Row heights are density-driven (session-row.tsx), and the preview window is
// defined in ROWS, not pixels — so it has to be measured, not hardcoded, or
// `comfortable`/`detailed` crop their third row mid-glyph.
const PREVIEW_ROW_GAP_PX = 1

export function previewWindowMaxHeight(
  density: SessionListDensity,
  rows = PROJECT_PREVIEW_COUNT,
  rowPx?: null | number
): string {
  const height = rowPx && rowPx > 0 ? rowPx : sessionRowEstimate(density)

  return `${rows * height + Math.max(0, rows - 1) * PREVIEW_ROW_GAP_PX}px`
}

/**
 * The preview window's height, in rows, measured off a real rendered row.
 *
 * `sessionRowEstimate` is what its name says — an estimate, sized for the
 * virtualizer, which only needs to be close. A hard `max-height` is stricter:
 * if a row ever renders TALLER than its estimate (OS font scaling, a label that
 * wraps), the cap reintroduces exactly the mid-glyph crop the density-driven
 * height exists to prevent. So measure the first row and size the window off
 * it, and keep the estimate only for the frame before that measurement lands.
 *
 * The read rides `useResizeObserver` (shared observer, post-layout timing)
 * rather than a synchronous layout effect — see that hook for why measuring
 * from a dirty commit thrashes layout.
 */
export function usePreviewWindowHeight(
  density: SessionListDensity,
  rowCount: number
): [RefObject<HTMLDivElement | null>, string] {
  const ref = useRef<HTMLDivElement>(null)
  // A measurement only describes the density and row set it was taken under.
  // Keying it means a density switch falls back to that density's estimate for
  // one frame instead of sizing the new rows with the old rows' height.
  const key = `${density}:${rowCount}`
  const [measured, setMeasured] = useState<{ key: string; rowPx: number } | null>(null)

  const measure = useCallback(() => {
    const first = ref.current?.firstElementChild

    if (!first) {
      return
    }

    const rowPx = Math.round(first.getBoundingClientRect().height)

    setMeasured(prev => (rowPx > 0 && (prev?.key !== key || prev.rowPx !== rowPx) ? { key, rowPx } : prev))
  }, [key])

  useResizeObserver(measure, ref)

  // Once the window is capped its own box stops changing, so row growth
  // underneath never reaches the observer. Re-measure whenever the key moves.
  useEffect(measure, [measure])

  return [ref, previewWindowMaxHeight(density, PROJECT_PREVIEW_COUNT, measured?.key === key ? measured.rowPx : null)]
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
