import { useStore } from '@nanostores/react'
import { useMemo } from 'react'

import type { SessionInfo } from '@/hermes'
import { $sidebarShowAllSessions } from '@/store/layout'
import { $projectOwnerBySessionId, $projects } from '@/store/projects'
import { $sessions } from '@/store/session'
import { $removedSessionIds } from '@/store/session-removal'
import { $sidebarSessionRankIds } from '@/store/sidebar-sort'

import { PROJECT_PREVIEW_COUNT } from './model'
import { overlayLivePreviews, sessionBucketId, type SidebarProjectTree } from './workspace-groups'

/** What the project rows drop: pins (they live in their own section) and anything the active filters
 *  exclude — the same predicate the previews below were built with, so a "Show all" hydration can't
 *  bring a row back. `counts` is how many of the backend's per-project totals it hides. */
export interface ProjectRowHidden {
  counts: Record<string, number>
  isHidden: (session: SessionInfo) => boolean
}

export interface ProjectRowData {
  hidden: ProjectRowHidden
  previews: Record<string, SessionInfo[]>
}

/**
 * The data a project row needs: the preview rows of every project (keyed by project id) and the
 * exclusion those previews were built with. A row hands both back to the view, so expanding it and
 * hydrating it ("Show all") agree with what the collapsed row already promised.
 *
 * The previews come from the whole tree, never just the overview list: the entered view draws the
 * projects nested under the one you are inside, where the overview list is empty. Keyed by id, the
 * entries for projects the overview does not list are inert there.
 */
export function useProjectRowData({
  isHiddenFromProjects,
  liveSessions,
  tree
}: {
  isHiddenFromProjects: (session: SessionInfo) => boolean
  /** Live `$sessions` of the profile on screen; a session the snapshot hasn't folded in yet still
   *  shows under its project (and carries its working arc). */
  liveSessions: SessionInfo[]
  /** The whole project tree; the entered view draws the projects nested under the one you are in. */
  tree?: SidebarProjectTree[]
}): ProjectRowData {
  const projects = useStore($projects)
  const owners = useStore($projectOwnerBySessionId)
  const sessions = useStore($sessions)
  const removedSessionIds = useStore($removedSessionIds)
  const sortOrderIds = useStore($sidebarSessionRankIds)
  const showAllSessions = useStore($sidebarShowAllSessions)

  const previews = useMemo<Record<string, SessionInfo[]>>(
    () =>
      overlayLivePreviews(tree ?? [], liveSessions, projects, showAllSessions ? Infinity : PROJECT_PREVIEW_COUNT, {
        removed: removedSessionIds,
        // Rank before the trim, so "3 priciest in this project" isn't "3 most recent, priciest first".
        rankIds: sortOrderIds
      }),
    [tree, liveSessions, projects, removedSessionIds, sortOrderIds, showAllSessions]
  )

  const hidden = useMemo<ProjectRowHidden>(() => {
    const isHidden = (session: SessionInfo) => isHiddenFromProjects(session) || removedSessionIds.has(session.id)
    const counts: Record<string, number> = {}

    for (const session of sessions) {
      const projectId = isHidden(session) ? sessionBucketId(session, projects, owners) : null

      if (projectId) {
        counts[projectId] = (counts[projectId] ?? 0) + 1
      }
    }

    return { isHidden, counts }
  }, [sessions, projects, owners, isHiddenFromProjects, removedSessionIds])

  return { hidden, previews }
}
