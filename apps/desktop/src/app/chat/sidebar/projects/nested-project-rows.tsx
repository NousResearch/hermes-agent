import { useStore } from '@nanostores/react'
import type * as React from 'react'

import type { NewSessionSplitHandler } from '@/app/chat/new-session-drag'
import type { SessionInfo } from '@/hermes'
import { $sidebarWorkspaceNodeOpen } from '@/store/layout'
import { $projectScope } from '@/store/project-scope'
import { $sessionDotStateById, rollupDotState } from '@/store/session-dot-state'

import {
  nestProjectsByParent,
  projectDepths,
  projectDescendantIds,
  projectSubtreeSessionIds,
  visibleProjectRows
} from './model'
import { ProjectOverviewRow } from './overview-row'
import type { SidebarProjectTree } from './workspace-groups'

/**
 * The projects nested under the one you are inside — the WHOLE subtree, parent-first, each row one
 * step further in than the project it nests under — drawn as doors: entering one makes it the
 * project you are inside, the same way (see `model.projectBackTarget` for the way back out).
 *
 * Each row carries its own preview rows and its own "Show all", so expanding it shows what is inside
 * without leaving the level; a collapsed project hides the rows under it, exactly as in the
 * overview. The caller renders these AFTER the entered project's own sessions, so a subproject reads
 * as sitting under its parent rather than above it.
 */
export function NestedProjectRows({
  hidden,
  onEnter,
  onNewSession,
  onNewSessionSplit,
  previews,
  projects = [],
  renderRows
}: {
  /** The exclusion `previews` was built with, reapplied when a row hydrates its own lanes. */
  hidden?: { counts: Record<string, number>; isHidden: (session: SessionInfo) => boolean }
  onEnter?: (id: string) => void
  onNewSession?: (path: null | string) => void
  onNewSessionSplit?: NewSessionSplitHandler
  /** Per-project preview rows from the backend tree, keyed by project id. */
  previews?: Record<string, SessionInfo[]>
  projects?: SidebarProjectTree[]
  renderRows?: (sessions: SessionInfo[]) => React.ReactNode
}) {
  const enteredId = useStore($projectScope)
  const dotStates = useStore($sessionDotStateById)
  const nodeOpen = useStore($sidebarWorkspaceNodeOpen)

  // Everything under the entered project, transitively: a nested project can hold another one, and
  // hiding that one behind a second drill-in hides work that is already on this level's list.
  const subtree = projects.filter(
    project => project.id !== enteredId && projectDescendantIds(projects, enteredId).has(project.id)
  )

  // Depth is read off the whole tree but drawn RELATIVE to the level you are standing on, so the
  // first row in sits one step in — the nesting belongs to the view you are in, not the model.
  const depths = projectDepths(projects)
  const enteredDepth = depths.get(enteredId) ?? 0
  const rows = visibleProjectRows(nestProjectsByParent(subtree), id => nodeOpen[id] ?? true)
  // A nested row that holds nothing but its own subprojects still needs its disclosure control —
  // same reason the overview reads this over the whole list rather than the visible rows.
  const nestedParentIds = new Set(projects.map(project => project.parentId).filter(Boolean))

  if (!rows.length) {
    return null
  }

  return (
    <>
      {rows.map(project => (
        <ProjectOverviewRow
          // The loudest status anywhere under that project, so a level down is visible without entering.
          attentionState={rollupDotState(dotStates, projectSubtreeSessionIds(projects, project.id))}
          depth={Math.max(1, (depths.get(project.id) ?? 0) - enteredDepth)}
          hasNestedProjects={nestedParentIds.has(project.id)}
          hiddenSessionCount={hidden?.counts[project.id]}
          isSessionHidden={hidden?.isHidden}
          key={project.id}
          onEnter={onEnter}
          onNewSession={onNewSession}
          onNewSessionSplit={onNewSessionSplit}
          previewSessions={previews?.[project.id]}
          project={project}
          renderRows={renderRows}
        />
      ))}
    </>
  )
}
