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
          setProject(next)
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
  }, [projectId, ready, treeRevision, scope, retryToken])

  return { project, failed, loading, retry: () => setRetryToken(token => token + 1) }
}
