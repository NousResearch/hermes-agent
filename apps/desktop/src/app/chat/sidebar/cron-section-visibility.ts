// Whether the sidebar's dedicated "Cron jobs" section should render.
//
// Deliberately independent of `worktreeGroupingActive` (Projects view): unlike
// agent sessions and messaging threads, a cron job has no project/worktree of
// its own to be grouped under, so hiding it in Projects view left no
// equivalent way to reach it there at all — the overlay (Cmd/Ctrl+K → "Cron")
// still worked, but the persistent sidebar entry point silently disappeared
// for anyone using Projects view.
export function shouldShowCronSection({
  cronJobsCount,
  trimmedQuery
}: {
  cronJobsCount: number
  trimmedQuery: string
}): boolean {
  return !trimmedQuery && cronJobsCount > 0
}
