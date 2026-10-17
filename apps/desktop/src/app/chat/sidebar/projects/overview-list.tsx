import type { useSensors } from '@dnd-kit/core'
import { useStore } from '@nanostores/react'
import type * as React from 'react'

import type { NewSessionSplitHandler } from '@/app/chat/new-session-drag'
import type { SessionInfo } from '@/hermes'
import { $sidebarShowAllSessions, $sidebarWorkspaceNodeOpen } from '@/store/layout'
import { $sessionDotStateById, rollupDotState } from '@/store/session-dot-state'

import { ReorderableList, useSortableBindings } from '../reorderable-list'

import { projectDepths, projectSubtreeSessionIds, visibleProjectRows } from './model'
import { ProjectOverviewRow } from './overview-row'
import type { ProjectNestPolicy } from './project-drag'
import type { SidebarProjectTree } from './workspace-groups'

interface ProjectOverviewListProps {
  activeProjectId?: null | string
  /** The section's dnd-kit sensors, so a project drag shares the sidebar's own activation rules. */
  dndSensors?: ReturnType<typeof useSensors>
  /** The nest/drop policy the section built (`createProjectNestResolver`), already wired to the store. */
  nest: ProjectNestPolicy
  onEnterProject?: (id: string) => void
  onNewSession?: (path: null | string) => void
  onNewSessionSplit?: NewSessionSplitHandler
  /** Drag-to-reorder the projects; absent leaves the rows static. */
  onReorderProjects?: (ids: string[]) => void
  /** The project tree in its model order (Home leads; see the section's caller). */
  projects: SidebarProjectTree[]
  /** Per-project preview rows from the backend tree, keyed by project id. */
  previews?: Record<string, SessionInfo[]>
  /** The exclusion `previews` was built with, reapplied when a row hydrates its own lanes. */
  hidden?: { counts: Record<string, number>; isHidden: (session: SessionInfo) => boolean }
  /** A project's placeholder rows while its lanes are only previewed. */
  renderPreviewRows: (items: SessionInfo[], projectId: string) => React.ReactNode
  renderRows: (sessions: SessionInfo[]) => React.ReactNode
}

/**
 * The project overview: every project as a row that drills into it AND reports the loudest status
 * anywhere under it — folded up from its own sessions and every nested project's, so a collapsed row
 * still says work is waiting inside. Home stays outside the sortable set: it is a fixture, not a
 * project to order.
 */
export function ProjectOverviewList({
  activeProjectId,
  dndSensors,
  hidden,
  nest,
  onEnterProject,
  onNewSession,
  onNewSessionSplit,
  onReorderProjects,
  previews,
  projects,
  renderPreviewRows,
  renderRows
}: ProjectOverviewListProps) {
  const dotStates = useStore($sessionDotStateById)
  const nodeOpen = useStore($sidebarWorkspaceNodeOpen)
  const showAllSessions = useStore($sidebarShowAllSessions)

  // The model is already ordered (Home leads; then the default sort groups
  // explicit-before-auto, with a manual drag-order winning when present).
  // Render in that order and make rows drag-to-reorder when a handler is
  // wired — Home stays outside the sortable list, it's a fixture.
  const home = projects[0]?.isNoProject ? projects[0] : undefined
  const sortableProjects = home ? projects.slice(1) : projects
  // A collapsed project hides its subprojects along with its sessions — the nest is a display
  // grouping, and a row nobody can see is not a row to render. Each project keeps its own open
  // flag, so whatever a subproject was left in survives its parent folding away and coming back.
  // The sortable ids stay whole: a drop resolves against the FULL order, so reordering while a
  // parent is closed cannot renumber the rows it hides.
  const visibleProjects = visibleProjectRows(sortableProjects, id => nodeOpen[id] ?? true)
  // A parent that folds its subprojects away cannot lose the control that opens it again, so this
  // reads the WHOLE list: `visibleProjects` no longer holds the rows a closed parent hides.
  const nestedParentIds = new Set(sortableProjects.map(project => project.parentId).filter(Boolean))
  // Each row's nesting depth, so a subproject of a subproject is drawn one step further in than the
  // subproject itself instead of beside it.
  const depths = projectDepths(projects)
  const projectsDraggable = sortableProjects.length > 1 && !!onReorderProjects
  const Row = projectsDraggable ? SortableProjectOverviewRow : ProjectOverviewRow

  const projectRow = (project: SidebarProjectTree, Component: typeof ProjectOverviewRow) => (
    <Component
      activeProjectId={activeProjectId}
      // The loudest status anywhere under this project, folded up from its own sessions and every
      // nested project's — a collapsed row still reports work waiting inside it.
      attentionState={rollupDotState(dotStates, projectSubtreeSessionIds(projects, project.id))}
      depth={depths.get(project.id) ?? 0}
      hasNestedProjects={nestedParentIds.has(project.id)}
      hiddenSessionCount={hidden?.counts[project.id]}
      isSessionHidden={hidden?.isHidden}
      key={project.id}
      onEnter={onEnterProject}
      onNewSession={onNewSession}
      onNewSessionSplit={onNewSessionSplit}
      // Keyed by project ID to match the producer: `overlayLivePreviews`
      // writes `out[node.id]` (workspace-groups.ts). A path key made Home
      // (path: null) and any id/path-divergent project fall back to stale
      // preview rows instead of the live overlay.
      previewSessions={previews?.[project.id]}
      project={project}
      renderRows={showAllSessions ? items => renderPreviewRows(items, project.id) : renderRows}
    />
  )

  const rows = visibleProjects.map(project => projectRow(project, Row))

  return (
    <>
      {home && projectRow(home, ProjectOverviewRow)}
      {projectsDraggable && onReorderProjects ? (
        <ReorderableList
          ids={sortableProjects.map(project => project.id)}
          onReorder={onReorderProjects}
          // The list's reflow follows the pointer's own crossings (see projects/project-drag.ts),
          // so a row cannot slide out from under a nest before the drop.
          resolveNest={nest.resolve}
          resolveSlot={nest.slot}
          sensors={dndSensors}
        >
          {rows}
        </ReorderableList>
      ) : (
        rows
      )}
    </>
  )
}

function SortableProjectOverviewRow(props: React.ComponentProps<typeof ProjectOverviewRow>) {
  return <ProjectOverviewRow {...props} {...useSortableBindings(props.project.id)} />
}
