import { useEffect, useState } from 'react'

import type { SidebarProjectTree } from '@/app/chat/sidebar/projects/workspace-groups'
import { fetchProjectSessions } from '@/store/projects'

interface ProjectSessionsState {
  failed: boolean
  hydrated: null | SidebarProjectTree
  projectId: string
}

/**
 * The selected project's fully hydrated lanes (`projects.project_sessions`).
 *
 * Not `supersedable`: that generation counter is shared with the sidebar's
 * drill-in, and a cockpit fetch must never cancel the sidebar's. Staleness is
 * guarded locally instead — a response for a project the user already left is
 * dropped. A failed refresh keeps the last good lanes (merge, don't clobber);
 * callers fall back to the tree's preview until the first answer lands.
 */
export function useProjectSessions(project: null | SidebarProjectTree): {
  failed: boolean
  hydrated: null | SidebarProjectTree
} {
  const [state, setState] = useState<null | ProjectSessionsState>(null)
  const projectId = project?.id ?? null
  // Refetch when the tree reports new activity for this project; the tree keeps
  // unchanged nodes by reference, so a no-op refresh doesn't refetch.
  const activityKey = project ? `${project.sessionCount}:${project.lastActive ?? 0}` : ''

  useEffect(() => {
    if (!projectId) {
      return
    }

    let cancelled = false

    const keepPrevious = (prev: null | ProjectSessionsState) => (prev?.projectId === projectId ? prev.hydrated : null)

    fetchProjectSessions(projectId, { supersedable: false }).then(
      hydrated => {
        if (!cancelled) {
          setState(prev => ({ failed: false, hydrated: hydrated ?? keepPrevious(prev), projectId }))
        }
      },
      () => {
        if (!cancelled) {
          setState(prev => ({ failed: true, hydrated: keepPrevious(prev), projectId }))
        }
      }
    )

    return () => {
      cancelled = true
    }
  }, [activityKey, projectId])

  const current = state?.projectId === projectId ? state : null

  return { failed: current?.failed ?? false, hydrated: current?.hydrated ?? null }
}
