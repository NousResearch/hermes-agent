import { atom } from 'nanostores'

import { getCronJobRuns } from '@/hermes'
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

// Stale-while-revalidate cache for cron run sessions. Sidebar peeks and the
// full overlay request different result windows, so each entry is scoped to a
// normalized (job id, limit) tuple. One entry also owns its in-flight refresh,
// preventing duplicate polls without allowing one result shape to overwrite
// another.
const DEFAULT_CRON_RUN_LIMIT = 20
const MAX_CRON_RUN_LIMIT = 100

type CronRunsCacheKey = readonly [jobId: string, limit: number]

interface CronRunsCacheEntry {
  inFlight?: Promise<SessionInfo[]>
  key: CronRunsCacheKey
  runs?: SessionInfo[]
}

const cronRunsCache = new Map<string, CronRunsCacheEntry>()

function cronRunsCacheKey(jobId: string, limit = DEFAULT_CRON_RUN_LIMIT): CronRunsCacheKey {
  const normalizedLimit = Number.isFinite(limit)
    ? Math.max(1, Math.min(Math.trunc(limit), MAX_CRON_RUN_LIMIT))
    : DEFAULT_CRON_RUN_LIMIT

  return [jobId, normalizedLimit]
}

export function getCachedCronRuns(jobId: string, limit = DEFAULT_CRON_RUN_LIMIT): SessionInfo[] | null {
  return cronRunsCache.get(JSON.stringify(cronRunsCacheKey(jobId, limit)))?.runs ?? null
}

export function loadCronJobRuns(jobId: string, limit = DEFAULT_CRON_RUN_LIMIT): Promise<SessionInfo[]> {
  const key = cronRunsCacheKey(jobId, limit)
  const cacheKey = JSON.stringify(key)
  const cached = cronRunsCache.get(cacheKey)

  if (cached?.inFlight) {
    return cached.inFlight
  }

  const entry = cached ?? { key }

  const request = getCronJobRuns(...key)
    .then(runs => {
      const current = cronRunsCache.get(cacheKey)

      if (current?.inFlight === request) {
        current.runs = runs
      }

      return runs
    })
    .finally(() => {
      const current = cronRunsCache.get(cacheKey)

      if (current?.inFlight !== request) {
        return
      }

      current.inFlight = undefined

      if (current.runs === undefined) {
        cronRunsCache.delete(cacheKey)
      }
    })

  entry.inFlight = request
  cronRunsCache.set(cacheKey, entry)

  return request
}

export function invalidateCronJobRuns(jobId: string): void {
  for (const [cacheKey, entry] of cronRunsCache) {
    if (entry.key[0] === jobId) {
      cronRunsCache.delete(cacheKey)
    }
  }
}
