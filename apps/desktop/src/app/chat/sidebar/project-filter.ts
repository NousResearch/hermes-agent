/**
 * Narrow the persisted sidebar project filter to ids the ACTIVE tree resolves.
 * The filter's storage is shared across profiles and outlives project
 * deletes/rebuilds, so ids that resolve nowhere must be inert, not fatal
 * (#96246, #97762). Fail-open while the tree is still loading.
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

/**
 * The sidebar overview is a session history, not a disk browser (#53004): an
 * auto-promoted repo with no sessions yet stays out of the sidebar until work
 * lands there (it reappears the moment it owns a session). Explicit projects
 * and the Home bucket always render — the user created or kept those. Session
 * counts come from the backend `projects.tree`, so this stays live without
 * client-side re-derivation.
 */
export function filterToSessionBearingProjects<
  T extends { isAuto?: boolean; isNoProject?: boolean; sessionCount?: number }
>(projects: readonly T[]): T[] {
  return projects.filter(project => !project.isAuto || project.isNoProject || (project.sessionCount ?? 0) > 0)
}
