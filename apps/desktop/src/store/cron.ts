import { atom, computed } from 'nanostores'

import { $profileScope, normalizeProfileKey } from '@/store/profile'
import type { CronJob } from '@/types/hermes'

export interface CronCacheRequest {
  generation: number
  invalidationEpoch: number
  scope: string
}

type CronCache = Record<string, CronJob[]>
type CronUpdater = CronJob[] | ((jobs: CronJob[]) => CronJob[])

// Jobs are backend-owned state. Keep every profile's list under an explicit UI
// scope rather than letting a late response overwrite one global display list.
export const $cronCache = atom<CronCache>({})

let invalidationEpoch = 0
const generationByScope = new Map<string, number>()

function normalizedScope(scope: string): string {
  return normalizeProfileKey(scope)
}

export function cronJobsForScope(scope: string, cache: CronCache = $cronCache.get()): CronJob[] {
  return cache[normalizedScope(scope)] ?? []
}

export const $cronJobs = computed([$cronCache, $profileScope], (cache, scope) => cronJobsForScope(scope, cache))

// Start every writer (fetch or mutation) in one generation domain so an older
// same-profile completion cannot clobber newer intent. The gateway epoch makes
// a same-named profile on a new backend a distinct identity.
export function cronCacheRequestForScope(scope: string): CronCacheRequest {
  const key = normalizedScope(scope)
  const generation = (generationByScope.get(key) ?? 0) + 1
  generationByScope.set(key, generation)

  return { generation, invalidationEpoch, scope: key }
}

export function cronCacheRequestIsCurrent(request: CronCacheRequest): boolean {
  const scope = normalizedScope(request.scope)

  return (
    request.invalidationEpoch === invalidationEpoch &&
    request.generation === generationByScope.get(scope) &&
    normalizedScope($profileScope.get()) === scope
  )
}

export function updateCronCacheForScope(request: CronCacheRequest, updater: CronUpdater): boolean {
  const scope = normalizedScope(request.scope)

  if (!cronCacheRequestIsCurrent(request)) {
    return false
  }

  const previous = cronJobsForScope(scope)
  const jobs = typeof updater === 'function' ? updater(previous) : updater
  const cache = $cronCache.get()

  if (cache[scope] !== jobs) {
    $cronCache.set({ ...cache, [scope]: jobs })
  }

  return true
}

export function invalidateCronCache(): void {
  invalidationEpoch += 1
  generationByScope.clear()
  $cronCache.set({})
}

// One-shot focus target: clicking "Manage" on a job sets this, then opens the
// cron overlay, which reads it once to select + scroll to that job. Cleared
// after consumption so re-opening cron normally doesn't re-focus a stale job.
export const $cronFocusJobId = atom<null | string>(null)
export const setCronFocusJobId = (id: null | string) => $cronFocusJobId.set(id)
