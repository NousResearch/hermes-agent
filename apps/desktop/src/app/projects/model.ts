import { sortProjectsForOverview } from '@/app/chat/sidebar/projects/model'
import {
  sessionRecency,
  type SidebarProjectTree,
  type SidebarSessionGroup,
  type SidebarWorkspaceTree
} from '@/app/chat/sidebar/projects/workspace-groups'
import type { ProjectInfo, SessionInfo } from '@/hermes'
import { normalize } from '@/lib/text'
import { filterVisibleProjects } from '@/store/layout'
import type { SessionDotState } from '@/store/session-dot-state'

import { PROJECTS_ROUTE } from '../routes'

// The Projects cockpit is a read-only lens over the SAME caches the sidebar
// paints from: `$projectTree` (backend `projects.tree`, the one membership
// authority), `$projects` (the `projects.list` rows that carry folders), and
// `projects.project_sessions` for the hydrated lanes. Nothing here decides
// membership or persists anything — these helpers only shape what is cached.

export const PROJECT_QUERY_PARAM = 'project'

export function projectOverviewRoute(projectId?: null | string): string {
  return projectId
    ? `${PROJECTS_ROUTE}?${new URLSearchParams({ [PROJECT_QUERY_PARAM]: projectId }).toString()}`
    : PROJECTS_ROUTE
}

/** Every visible real project in the tree, in the sidebar overview's order.
 * The Home bucket, archived projects, and dismissed auto-discovered repos are
 * not projects the user can operate on, so they stay out. */
export function cockpitProjects(
  tree: SidebarProjectTree[],
  activeProjectId: null | string,
  dismissedAutoProjectIds: readonly string[] = []
): SidebarProjectTree[] {
  return sortProjectsForOverview(
    filterVisibleProjects(tree, dismissedAutoProjectIds).filter(project => !project.isNoProject && !project.archived),
    activeProjectId
  )
}

export function filterCockpitProjects(projects: SidebarProjectTree[], query: string): SidebarProjectTree[] {
  const q = normalize(query)

  if (!q) {
    return projects
  }

  return projects.filter(
    project => normalize(project.label).includes(q) || normalize(project.path ?? '').includes(q)
  )
}

export interface ProjectFolderFact {
  isPrimary: boolean
  label: null | string
  path: string
}

/** Folders for the detail view: the explicit project's own folder rows (primary
 *  first) when `projects.list` has them, else the tree node's root folder. */
export function projectFolders(project: SidebarProjectTree, info?: ProjectInfo): ProjectFolderFact[] {
  if (info?.folders.length) {
    return [...info.folders]
      .sort((a, b) => Number(b.is_primary) - Number(a.is_primary))
      .map(folder => ({ isPrimary: folder.is_primary, label: folder.label, path: folder.path }))
  }

  const root = project.path?.trim()

  return root ? [{ isPrimary: true, label: null, path: root }] : []
}

export function projectPrimaryPath(project: SidebarProjectTree, info?: ProjectInfo): null | string {
  return (
    info?.primary_path?.trim() ||
    project.path?.trim() ||
    project.repos.find(repo => repo.path)?.path?.trim() ||
    null
  )
}

export type LaneKind = 'kanban' | 'main' | 'worktree'

export function laneKind(group: SidebarSessionGroup): LaneKind {
  if (group.isKanban) {
    return 'kanban'
  }

  return group.isMain || group.isHome ? 'main' : 'worktree'
}

/** Repos with at least one git lane. A plain folder's heuristic lane
 *  (`isGit === false`) has no branch to report — it is already under Folders. */
export function gitRepositories(project: SidebarProjectTree): SidebarWorkspaceTree[] {
  return project.repos
    .map(repo => ({ ...repo, groups: repo.groups.filter(group => group.isGit !== false) }))
    .filter(repo => repo.groups.length > 0)
}

/** The project's sessions, newest first. Prefers the hydrated lanes (every
 *  session the backend assigned, manual moves included) over the tree's short
 *  preview; tombstoned ids stay hidden while their delete is in flight. */
export function projectSessionList(
  project: SidebarProjectTree,
  hydrated: null | SidebarProjectTree,
  removedIds: ReadonlySet<string> = new Set()
): SessionInfo[] {
  const source = hydrated
    ? hydrated.repos.flatMap(repo => repo.groups.flatMap(group => group.sessions))
    : (project.previewSessions ?? [])

  const byId = new Map<string, SessionInfo>()

  for (const session of source) {
    if (!removedIds.has(session.id) && !byId.has(session.id)) {
      byId.set(session.id, session)
    }
  }

  return [...byId.values()].sort((a, b) => sessionRecency(b) - sessionRecency(a))
}

export type LiveSessionState = 'background' | 'needs-input' | 'stalled' | 'working'

const LIVE_STATES: ReadonlySet<SessionDotState> = new Set<LiveSessionState>([
  'background',
  'needs-input',
  'stalled',
  'working'
])

/** A session's live state, or null when nothing is running. Idle, draft and
 *  unread report no activity — the cockpit never invents one. */
export function liveSessionState(state: SessionDotState | undefined): LiveSessionState | null {
  return state && LIVE_STATES.has(state) ? (state as LiveSessionState) : null
}

export interface ActiveProjectSession {
  session: SessionInfo
  state: LiveSessionState
}

export function splitActiveSessions(
  sessions: SessionInfo[],
  dotStates: Readonly<Record<string, SessionDotState>>
): { active: ActiveProjectSession[]; recent: SessionInfo[] } {
  const active: ActiveProjectSession[] = []
  const recent: SessionInfo[] = []

  for (const session of sessions) {
    const state = liveSessionState(dotStates[session.id])

    if (state) {
      active.push({ session, state })
    } else {
      recent.push(session)
    }
  }

  return { active, recent }
}
