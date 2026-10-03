import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import type { SidebarProjectTree } from '@/app/chat/sidebar/projects/workspace-groups'
import { $profileScope, ALL_PROFILES } from '@/store/profile'
import { $projectsOwnerKey, fetchProjectSessions } from '@/store/projects'

/**
 * - `preview`: nothing hydrated for this owner yet; the tree's preview stands in.
 * - `hydrated`: the backend's complete lanes for this owner and project.
 * - `failed`: the read failed (or answered nothing); `hydrated` is the last
 *   good answer for the SAME owner and project, if any.
 * - `limited`: All Profiles has no single backend to hydrate one project from,
 *   so the session list is knowingly incomplete and must say so.
 */
export type ProjectSessionsStatus = 'failed' | 'hydrated' | 'limited' | 'preview'

export interface ProjectSessionsResult {
  hydrated: null | SidebarProjectTree
  status: ProjectSessionsStatus
}

interface ProjectSessionsState {
  failed: boolean
  hydrated: null | SidebarProjectTree
  key: string
}

/**
 * The selected project's fully hydrated lanes (`projects.project_sessions`).
 *
 * State and in-flight reads are keyed by owner (connection + profile view) AND
 * project, so a switch away and back (A → B → A) never paints another owner's
 * sessions, and a late answer for a departed owner or project is dropped.
 *
 * Not `supersedable`: that generation counter is shared with the sidebar's
 * drill-in, and a cockpit fetch must never cancel the sidebar's. A failed
 * refresh keeps the last good lanes for the same owner (merge, don't clobber);
 * callers fall back to the tree's preview until the first answer lands.
 * `refreshToken` lets an explicit Refresh re-read even when the tree reports no
 * new activity (a retry after failure, or a title-only change).
 */
export function useProjectSessions(project: null | SidebarProjectTree, refreshToken = 0): ProjectSessionsResult {
  // The connection + profile view the id belongs to. Project ids repeat across
  // profiles (auto-projects are folder paths), so an id alone never names
  // whose sessions these are.
  const owner = useStore($projectsOwnerKey)
  const allProfiles = useStore($profileScope) === ALL_PROFILES
  const [state, setState] = useState<null | ProjectSessionsState>(null)
  const projectId = project?.id ?? null
  const key = projectId ? `${owner}\u0000${projectId}` : null
  // Refetch when the tree reports new activity for this project; the tree keeps
  // unchanged nodes by reference, so a no-op refresh doesn't refetch.
  const activityKey = project ? `${project.sessionCount}:${project.lastActive ?? 0}` : ''

  useEffect(() => {
    if (!key || !projectId || allProfiles) {
      return
    }

    let cancelled = false

    const keepPrevious = (prev: null | ProjectSessionsState) => (prev?.key === key ? prev.hydrated : null)

    fetchProjectSessions(projectId, { supersedable: false }).then(
      hydrated => {
        if (!cancelled) {
          // `null` is no answer (owner moved mid-read, or the project is gone):
          // never a complete, empty hydration.
          setState(prev =>
            hydrated ? { failed: false, hydrated, key } : { failed: true, hydrated: keepPrevious(prev), key }
          )
        }
      },
      () => {
        if (!cancelled) {
          setState(prev => ({ failed: true, hydrated: keepPrevious(prev), key }))
        }
      }
    )

    return () => {
      cancelled = true
    }
  }, [activityKey, allProfiles, key, projectId, refreshToken])

  if (!key) {
    return { hydrated: null, status: 'preview' }
  }

  if (allProfiles) {
    return { hydrated: null, status: 'limited' }
  }

  const current = state?.key === key ? state : null

  if (!current) {
    return { hydrated: null, status: 'preview' }
  }

  return { hydrated: current.hydrated, status: current.failed ? 'failed' : 'hydrated' }
}
