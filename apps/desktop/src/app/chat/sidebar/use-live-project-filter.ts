import { useStore } from '@nanostores/react'
import { useMemo } from 'react'

import { $sidebarProjectFilter } from '@/store/layout'
import { $projectTreeIds } from '@/store/projects'

import { resolveLiveProjectFilter } from './project-filter'

/**
 * The sidebar's project filter as the ACTIVE tree can honour it, and the ids of
 * the buckets that tree built.
 *
 * The persisted filter's storage is shared across profiles, so ids picked in
 * another profile don't resolve in the active one and the raw membership
 * whitelist empties every tier of the sidebar (#96246). Narrow to ids the
 * active tree resolves; dead ids are inert, not fatal.
 *
 * The tree ids let the filter judge a row past the tree's session window: a cwd
 * walk that names no bucket the tree built files under Home, where the backend
 * puts it, instead of under an invented auto project no filter offers (#159024).
 */
export function useLiveProjectFilter(tree: readonly { id: string }[] | null | undefined): {
  projectFilter: readonly string[]
  projectTreeIds: ReadonlySet<string>
} {
  const persisted = useStore($sidebarProjectFilter)
  const projectTreeIds = useStore($projectTreeIds)
  const projectFilter = useMemo(() => resolveLiveProjectFilter(persisted, tree), [persisted, tree])

  return { projectFilter, projectTreeIds }
}
