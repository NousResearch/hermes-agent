import type * as React from 'react'

import type { NewSessionSplitHandler } from '@/app/chat/new-session-drag'
import type { HermesGitWorktree } from '@/global'
import type { SessionInfo } from '@/hermes'

import { EnteredProjectContent } from './entered-content'
import { NestedProjectRows } from './nested-project-rows'
import type { SidebarProjectTree } from './workspace-groups'

// The entered project's level: the way back out, the project's own sessions —
// or the caller's empty state, so the pane is never a bare spinner or blank
// while lanes hydrate — and then the projects nested under this one, each a row
// that shows what it holds and can be drilled a level further down. The nested
// rows come last so they sit under the sessions of the project they belong to.
export function EnteredProjectView({
  backRow,
  emptyState,
  hasContent,
  liveSessions,
  nestedHidden,
  nestedPreviews,
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
  /** Exclusions a nested row's "Show all" must share with the previews. */
  nestedHidden?: { counts: Record<string, number>; isHidden: (session: SessionInfo) => boolean }
  /** Preview rows per project id, so a nested row expands to what it holds. */
  nestedPreviews?: Record<string, SessionInfo[]>
  /** The overview tree this project came from, for the rows nested under it. */
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
      <NestedProjectRows
        hidden={nestedHidden}
        onEnter={onEnterProject}
        onNewSession={onNewSession}
        onNewSessionSplit={onNewSessionSplit}
        previews={nestedPreviews}
        projects={nestedProjects}
        renderRows={renderRows}
      />
    </>
  )
}
