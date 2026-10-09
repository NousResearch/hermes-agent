import { useStore } from '@nanostores/react'

import type { NewSessionSplitHandler } from '@/app/chat/new-session-drag'
import { $projectScope } from '@/store/project-scope'
import { $sessionDotStateById, rollupDotState } from '@/store/session-dot-state'

import { projectSubtreeSessionIds } from './model'
import { ProjectOverviewRow } from './overview-row'
import type { SidebarProjectTree } from './workspace-groups'

/**
 * The projects nested directly under the one you are inside — the same rows the overview nests
 * beneath it, in the order that list was left in — drawn as doors one level further down: entering
 * one shows ITS children here, the same way (see `model.projectBackTarget` for the way back out).
 */
export function NestedProjectRows({
  onEnter,
  onNewSession,
  onNewSessionSplit,
  projects = []
}: {
  projects?: SidebarProjectTree[]
  onEnter?: (id: string) => void
  onNewSession?: (path: null | string) => void
  onNewSessionSplit?: NewSessionSplitHandler
}) {
  const enteredId = useStore($projectScope)
  const dotStates = useStore($sessionDotStateById)
  const nested = projects.filter(project => project.parentId === enteredId)

  if (!nested.length) {
    return null
  }

  return (
    <>
      {nested.map(project => (
        <ProjectOverviewRow
          // The loudest status anywhere under that project, so a level down is visible without entering.
          attentionState={rollupDotState(dotStates, projectSubtreeSessionIds(projects, project.id))}
          key={project.id}
          onEnter={onEnter}
          onNewSession={onNewSession}
          onNewSessionSplit={onNewSessionSplit}
          project={project}
        />
      ))}
    </>
  )
}
