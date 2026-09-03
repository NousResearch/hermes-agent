import { atom } from 'nanostores'

import type { CronJob } from '@/types/hermes'

// Cron *jobs* (not run sessions) power the sidebar "Cron jobs" section. Listing
// the job — schedule, state, live next-run countdown — makes the job the
// first-class entity; its runs (sessions) resolve under it in the cron detail.
export const $cronJobs = atom<CronJob[]>([])

type CronJobsCache = Record<string, CronJob[]>

export interface CronJobsRequest {
  epoch: number
  generation: number
  scope: string
}

export interface CronJobsScopeToken {
  epoch: number
  generation: number
  scope: string
}

let cronJobsCache: CronJobsCache = {}
let cronJobsInvalidationEpoch = 0
let activeCronJobsScope = ''
const requestGenerationByScope = new Map<string, number>()
const actionGenerationByScope = new Map<string, number>()

function activateCronJobsScope(scope: string): void {
  if (scope === activeCronJobsScope) {
    return
  }

  activeCronJobsScope = scope
  $cronJobs.set(cronJobsCache[scope] ?? [])
}

function nextGeneration(generations: Map<string, number>, scope: string): number {
  const generation = (generations.get(scope) ?? 0) + 1
  generations.set(scope, generation)

  return generation
}

export function beginCronJobsRequest(scope: string): CronJobsRequest {
  activateCronJobsScope(scope)

  return {
    epoch: cronJobsInvalidationEpoch,
    generation: nextGeneration(requestGenerationByScope, scope),
    scope
  }
}

export function beginCronJobsAction(scope: string): CronJobsScopeToken {
  activateCronJobsScope(scope)

  return {
    epoch: cronJobsInvalidationEpoch,
    generation: nextGeneration(actionGenerationByScope, scope),
    scope
  }
}

export function isCronJobsScopeCurrent(token: CronJobsScopeToken): boolean {
  return (
    token.epoch === cronJobsInvalidationEpoch &&
    token.scope === activeCronJobsScope &&
    token.generation === actionGenerationByScope.get(token.scope)
  )
}

export function isCronJobsRequestCurrent(request: CronJobsRequest): boolean {
  return (
    request.epoch === cronJobsInvalidationEpoch &&
    request.scope === activeCronJobsScope &&
    request.generation === requestGenerationByScope.get(request.scope)
  )
}

// A gateway switch can retain the same profile name. Clearing per-scope
// generations alone would let a pre-switch completion reuse generation 1, so
// every token also carries this monotonic backend epoch.
export function invalidateCronJobsRequests(): void {
  cronJobsInvalidationEpoch += 1
  requestGenerationByScope.clear()
  actionGenerationByScope.clear()
  cronJobsCache = {}
  $cronJobs.set([])
}

export function commitCronJobsRequest(request: CronJobsRequest, jobs: CronJob[]): boolean {
  if (!isCronJobsRequestCurrent(request)) {
    return false
  }

  // Consume the token so neither a duplicate completion nor any older request
  // can publish after this authoritative snapshot.
  nextGeneration(requestGenerationByScope, request.scope)
  cronJobsCache = { ...cronJobsCache, [request.scope]: jobs }
  $cronJobs.set(jobs)

  return true
}

export const setCronJobs = (jobs: CronJob[]) => {
  nextGeneration(requestGenerationByScope, activeCronJobsScope)
  cronJobsCache = { ...cronJobsCache, [activeCronJobsScope]: jobs }
  $cronJobs.set(jobs)
}

// In-place edit so the cron overlay's mutations (create/edit/delete/pause/…)
// land in the same atom the sidebar renders — no stale list until the next poll.
export const updateCronJobs = (fn: (jobs: CronJob[]) => CronJob[]) => {
  nextGeneration(requestGenerationByScope, activeCronJobsScope)
  const jobs = fn($cronJobs.get())
  cronJobsCache = { ...cronJobsCache, [activeCronJobsScope]: jobs }
  $cronJobs.set(jobs)
}

// One-shot focus target: clicking "Manage" on a job sets this, then opens the
// cron overlay, which reads it once to select + scroll to that job. Cleared
// after consumption so re-opening cron normally doesn't re-focus a stale job.
export const $cronFocusJobId = atom<null | string>(null)
export const setCronFocusJobId = (id: null | string) => $cronFocusJobId.set(id)

// Shell-owned one-shot intent for stores without router context. Do not set a
// focus id here: the cron overlay's first fetch may not have loaded that row.
export const $cronReviewRequest = atom(0)
export const requestCronReview = () => $cronReviewRequest.set($cronReviewRequest.get() + 1)
