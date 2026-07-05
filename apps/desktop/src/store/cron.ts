import { atom } from 'nanostores'

import type { CronJob, SessionInfo } from '@/types/hermes'

// Cron *jobs* (not run sessions) power the sidebar "Cron jobs" section. Listing
// the job — schedule, state, live next-run countdown — makes the job the
// first-class entity; its runs (sessions) resolve under it in the cron detail.
export const $cronJobs = atom<CronJob[]>([])
export const setCronJobs = (jobs: CronJob[]) => $cronJobs.set(jobs)

// In-place edit so the cron overlay's mutations (create/edit/delete/pause/…)
// land in the same atom the sidebar renders — no stale list until the next poll.
export const updateCronJobs = (fn: (jobs: CronJob[]) => CronJob[]) => $cronJobs.set(fn($cronJobs.get()))

// One-shot focus target: clicking "Manage" on a job sets this, then opens the
// cron overlay, which reads it once to select + scroll to that job. Cleared
// after consumption so re-opening cron normally doesn't re-focus a stale job.
export const $cronFocusJobId = atom<null | string>(null)
export const setCronFocusJobId = (id: null | string) => $cronFocusJobId.set(id)

// Stale-while-revalidate cache for cron run sessions. Unlike $cronJobs (a
// reactive atom shared by sidebar + overlay), runs are per-job and read-only —
// no cross-component sync needed. The cache just prevents a remount spinner
// when the panel is closed and reopened: the component renders stale data
// instantly, then the background poll silently replaces it with fresh results.
const cronRunsCache = new Map<string, SessionInfo[]>()

export function getCachedCronRuns(jobId: string): SessionInfo[] | null {
  return cronRunsCache.get(jobId) ?? null
}

export function setCachedCronRuns(jobId: string, runs: SessionInfo[]): void {
  cronRunsCache.set(jobId, runs)
}
