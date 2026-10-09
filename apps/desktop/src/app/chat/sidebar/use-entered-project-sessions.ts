import { replaceEqualDeep } from '@tanstack/react-query'
import { useEffect, useState } from 'react'

import { fetchProjectSessions } from '@/store/projects'

import type { SidebarProjectTree } from './projects/workspace-groups'

// The mounted drill-in owns its outcome. A global error flag lets a departed
// project's slow failure overwrite the next project's successful load.
export function useEnteredProjectSessions(
  projectId: string | undefined,
  ready: boolean,
  treeRevision: readonly SidebarProjectTree[],
  scope: string
) {
  const [project, setProject] = useState<SidebarProjectTree | null>(null)
  const [failed, setFailed] = useState(false)
  const [loading, setLoading] = useState(false)
  const [retryToken, setRetryToken] = useState(0)

  // Refetch when the entered project's own overview node changes, not on every
  // tree refresh: each `projects.project_sessions` call hydrates the whole tree
  // on the backend, which takes seconds over a remote gateway (#77591). The
  // tree keeps unchanged nodes by reference, so this is stable across no-ops.
  const enteredNode = projectId ? treeRevision.find(node => node.id === projectId) : undefined

  useEffect(() => {
    setProject(null)
  }, [projectId, scope])

  useEffect(() => {
    let cancelled = false
    setFailed(false)

    if (!projectId || !ready) {
      setProject(null)
      setLoading(false)

      return
    }

    setLoading(true)
    void fetchProjectSessions(projectId)
      .then(next => {
        if (!cancelled) {
          // An unchanged answer keeps its reference, so the lanes don't rebuild.
          setProject(current => replaceEqualDeep(current, next))
        }
      })
      .catch(() => {
        if (!cancelled) {
          setFailed(true)
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(false)
        }
      })

    return () => {
      cancelled = true
    }
  }, [projectId, ready, enteredNode, scope, retryToken])

  // A background refetch keeps painting the rows it has; only a drill-in with
  // nothing loaded yet reports loading (the sidebar shows skeletons for it).
  return { project, failed, loading: loading && !project, retry: () => setRetryToken(token => token + 1) }
}
