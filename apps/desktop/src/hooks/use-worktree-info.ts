import { useEffect, useMemo, useRef, useState } from 'react'

import { uniqueCwds, type WorktreeResolver } from '@/app/chat/sidebar/workspace-groups'
import type { HermesWorktreeInfo } from '@/global'
import type { SessionInfo } from '@/hermes'
import { desktopFsCacheKey, desktopWorktrees } from '@/lib/desktop-fs'

type WorktreeMap = Record<string, HermesWorktreeInfo | null>

/**
 * Probe the local filesystem for the git-worktree identity of each session cwd
 * and return a resolver the grouping uses to build `parent → worktree`. Results
 * are cached per cwd (and reset when the backend connection changes), so a probe
 * runs once per directory. Unresolved cwds (probe pending, remote backend, or
 * non-git dirs) fall back to the path-name heuristic in `workspaceTreeFor`.
 */
export function useWorktreeInfo(sessions: SessionInfo[], enabled: boolean): WorktreeResolver {
  const [map, setMap] = useState<WorktreeMap>({})
  const cacheRef = useRef<{ data: WorktreeMap; key: string }>({ data: {}, key: '' })

  useEffect(() => {
    if (!enabled) {
      return
    }

    const key = desktopFsCacheKey()

    if (cacheRef.current.key !== key) {
      cacheRef.current = { data: {}, key }
      setMap({})
    }

    const missing = uniqueCwds(sessions).filter(cwd => !(cwd in cacheRef.current.data))

    if (!missing.length) {
      return
    }

    let cancelled = false

    void desktopWorktrees(missing)
      .then(result => {
        if (cancelled) {
          return
        }

        // Record every probed cwd (null when absent) so we never re-probe it.
        const next: WorktreeMap = { ...cacheRef.current.data }

        for (const cwd of missing) {
          next[cwd] = result[cwd] ?? null
        }

        cacheRef.current = { data: next, key }
        setMap(next)
      })
      .catch(() => {
        // Bridge unavailable / probe failed — leave cwds unresolved so the
        // heuristic fallback handles them.
      })

    return () => {
      cancelled = true
    }
  }, [sessions, enabled])

  return useMemo<WorktreeResolver>(() => (cwd: string) => map[cwd], [map])
}
