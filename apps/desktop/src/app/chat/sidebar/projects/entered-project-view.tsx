import type * as React from 'react'

import type { NewSessionSplitHandler } from '@/app/chat/new-session-drag'
import type { HermesGitWorktree } from '@/global'
import type { SessionInfo } from '@/hermes'

import { EnteredProjectContent } from './entered-content'
import { NestedProjectRows } from './nested-project-rows'
import type { SidebarProjectTree } from './workspace-groups'

// The entered project's level: the way back out, the projects nested directly
// under this one (each a row that drills a level further down), then the
// project's own sessions — or the caller's empty state, so the pane is never a
// bare spinner or blank while lanes hydrate.
export function EnteredProjectView({
  backRow,
  emptyState,
  hasContent,
  liveSessions,
  nestedProjects,
  onEnterProject,
  onNewSession,
  onNewSessionSplit,
  project,
  removedSessionIds,
  renderRows,
  repoWorktrees
}: {
  /** The "back" row: up to the parent project, or out to the overview. */
  backRow?: React.ReactNode
  emptyState: React.ReactNode
  /** Whether the project has sessions or declared repos of its own to show. */
  hasContent: boolean
  liveSessions?: SessionInfo[]
  /** The overview tree this project came from, for the rows nested directly under it. */
  nestedProjects?: SidebarProjectTree[]
  onEnterProject?: (id: string) => void
  onNewSession?: (path: null | string) => void
  onNewSessionSplit?: NewSessionSplitHandler
  project: SidebarProjectTree
  removedSessionIds?: ReadonlySet<string>
  renderRows: (sessions: SessionInfo[]) => React.ReactNode
  repoWorktrees?: Record<string, HermesGitWorktree[]>
}) {
  return (
    <>
      {backRow}
      <NestedProjectRows
        onEnter={onEnterProject}
        onNewSession={onNewSession}
        onNewSessionSplit={onNewSessionSplit}
        projects={nestedProjects}
      />
      {hasContent ? (
        <EnteredProjectContent
          liveSessions={liveSessions}
          onNewSession={onNewSession}
          onNewSessionSplit={onNewSessionSplit}
          project={project}
          removedSessionIds={removedSessionIds}
          renderRows={renderRows}
          repoWorktrees={repoWorktrees}
        />
      ) : (
        emptyState
      )}
    </>
  )
}
