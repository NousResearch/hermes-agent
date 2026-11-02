import type { SessionInfo } from '@/types/hermes'

import { filterToSessionBearingProjects } from '../project-filter'

import { orderProjectsByIds, sortProjectsAlphabetically, sortProjectsForOverview } from './model'
import { excludeProjectSessions, type SidebarProjectTree, type SidebarWorkspaceTree } from './workspace-groups'

interface ProjectOverviewInput {
  activeProjectId: null | string
  // The opt-in "Sort sidebar alphabetically" preference.
  alphabetical: boolean
  homeLabel: string
  isHidden: (session: SessionInfo) => boolean
  orderRepos: (repos: SidebarWorkspaceTree[]) => SidebarWorkspaceTree[]
  projectFilter: readonly string[]
  projectOrderIds: string[]
  // The backend tree with dismissed auto-projects already removed.
  projects: readonly SidebarProjectTree[]
}

/**
 * The sidebar's top-level project list: the backend tree narrowed to what the
 * user wants to see, sorted, then the user's manual drag-order layered on top.
 * Alphabetical mode sorts Home-first A→Z and ignores the saved drag-order (it
 * stays stored and applies again once the preference is switched off).
 */
export function buildProjectOverview({
  activeProjectId,
  alphabetical,
  homeLabel,
  isHidden,
  orderRepos,
  projectFilter,
  projectOrderIds,
  projects
}: ProjectOverviewInput): SidebarProjectTree[] {
  const visible = filterToSessionBearingProjects(projects)
    // A filtered-out project drops its whole lane, header included — hiding
    // only its rows would leave a row of empty folders behind.
    .filter(project => !projectFilter.length || projectFilter.includes(project.id))
    .map(project =>
      excludeProjectSessions(
        {
          ...project,
          // Home is synthetic, so its name is ours to translate — every
          // other label is a repo basename or a name the user typed.
          label: project.isNoProject ? homeLabel : project.label,
          repos: orderRepos(project.repos)
        },
        isHidden
      )
    )

  const sorted = alphabetical ? sortProjectsAlphabetically(visible) : sortProjectsForOverview(visible, activeProjectId)

  // Layer the user's manual drag-order on top of the deterministic sort. Empty
  // (default) returns `sorted` untouched; projects the user hasn't ordered yet
  // keep their sorted position rather than jumping the hand-picked list.
  // Deduplicate by id — safety net for backend merge edge cases
  // (e.g. Windows path-casing differences across profiles).
  const seen = new Set<string>()

  const deduped = sorted.filter(project => {
    if (seen.has(project.id)) {
      return false
    }

    seen.add(project.id)

    return true
  })

  return alphabetical ? deduped : orderProjectsByIds(deduped, projectOrderIds)
}
