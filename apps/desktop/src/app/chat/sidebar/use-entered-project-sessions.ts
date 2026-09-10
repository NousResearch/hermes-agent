import { useEffect, useState } from 'react'

import { fetchProjectSessions, ProjectSessionsSuperseded } from '@/store/projects'

import type { SidebarProjectTree } from './projects/workspace-groups'

// A superseded answer is not an answer. Committing it as `null` painted the
// entered project as empty — the overview node's lanes carry no rows by design —
// which is how a drill-in ended up showing branch headers with nothing under
// them. The retry budget is spent inside the same effect (the effect also re-runs
// on every tree revision, so a live sidebar re-asks anyway); once it is spent the
// read reports FAILURE, which the sidebar renders as its Retry affordance
// instead of a project that looks empty.
const SUPERSEDED_RETRY_LIMIT = 3
const SUPERSEDED_RETRY_MS = 150

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
    let attempts = 0
    let timer: undefined | number
    setFailed(false)

    if (!projectId || !ready) {
      setProject(null)
      setLoading(false)

      return
    }

    setLoading(true)

    const run = () => {
      void fetchProjectSessions(projectId)
        .then(next => {
          if (!cancelled) {
            setProject(next)
          }
        })
        .catch(error => {
          if (cancelled) {
            return
          }

          if (error instanceof ProjectSessionsSuperseded && attempts < SUPERSEDED_RETRY_LIMIT) {
            attempts += 1
            timer = window.setTimeout(() => {
              timer = undefined
              run()
            }, SUPERSEDED_RETRY_MS)

            return
          }

          setFailed(true)
        })
        .finally(() => {
          // A scheduled retry keeps the pending state: the project is still
          // being read, so nothing may render as its (empty) content yet.
          if (!cancelled && timer === undefined) {
            setLoading(false)
          }
        })
    }

    run()

    return () => {
      cancelled = true

      if (timer !== undefined) {
        window.clearTimeout(timer)
      }
    }
  }, [projectId, ready, treeRevision, scope, retryToken])

  return { project, failed, loading, retry: () => setRetryToken(token => token + 1) }
}
