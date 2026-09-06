/**
 * Project-filter resolution against the ACTIVE profile's project tree.
 *
 * The sidebar's project filter is a `persistentAtom` whose storage key is
 * shared across profiles. Ids picked in profile B (explicit `p_<hex>` ids or
 * repo-root paths) do not resolve against profile A's per-profile
 * `projects.db`, and both application sites treat the raw filter as a
 * membership whitelist — so one cross-profile switch empties every tier of
 * the sidebar, reading as data loss until the filter is manually cleared
 * (#96246).
 *
 * `resolveLiveProjectFilter` narrows the persisted filter to ids the given
 * tree actually resolves. Ids that resolve nowhere become inert (the filter
 * stops narrowing) instead of fatal (everything filters out). While the tree
 * is still loading (empty), an explicit non-empty filter resolves to an
 * empty live set — fail-open, matching the tree-refresh re-hydration timing.
 */
export function resolveLiveProjectFilter(
  projectFilter: readonly string[],
  tree: readonly { id: string }[] | null | undefined
): readonly string[] {
  if (!projectFilter.length) {
    return projectFilter
  }

  if (!tree || !tree.length) {
    return []
  }

  const liveIds = new Set(tree.map(project => project.id))

  return projectFilter.filter(id => liveIds.has(id))
}
