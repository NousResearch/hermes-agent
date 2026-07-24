import { atom } from 'nanostores'

import type { CronJob } from '@/types/hermes'

// Cron *jobs* (not run sessions) power the sidebar "Cron jobs" section. Listing
// the job — schedule, state, live next-run countdown — makes the job the
// first-class entity; its runs (sessions) resolve under it in the cron detail.
export const $cronJobs = atom<CronJob[]>([])

export interface CronJobsRequest {
  generation: number
  scope: string
}

export interface CronJobsScopeToken {
  generation: number
  scope: string
}

let cronJobsRequestGeneration = 0
let cronJobsRequestScope = ''
let cronJobsScopeGeneration = 0

function activateCronJobsScope(scope: string): void {
  if (scope === cronJobsRequestScope) {
    return
  }

  cronJobsRequestScope = scope
  cronJobsRequestGeneration += 1
  cronJobsScopeGeneration += 1
}

export function beginCronJobsRequest(scope: string): CronJobsRequest {
  activateCronJobsScope(scope)
  cronJobsRequestGeneration += 1

  return { generation: cronJobsRequestGeneration, scope }
}

export function beginCronJobsAction(scope: string): CronJobsScopeToken {
  activateCronJobsScope(scope)

  return { generation: cronJobsScopeGeneration, scope }
}

export function isCronJobsScopeCurrent(token: CronJobsScopeToken): boolean {
  return token.scope === cronJobsRequestScope && token.generation === cronJobsScopeGeneration
}

export function isCronJobsRequestCurrent(request: CronJobsRequest): boolean {
  return request.scope === cronJobsRequestScope && request.generation === cronJobsRequestGeneration
}

export function invalidateCronJobsRequests(): void {
  cronJobsRequestGeneration += 1
  cronJobsScopeGeneration += 1
}

export function commitCronJobsRequest(request: CronJobsRequest, jobs: CronJob[]): boolean {
  if (!isCronJobsRequestCurrent(request)) {
    return false
  }

  // Consume the token so neither a duplicate completion nor any older request
  // can publish after this authoritative snapshot.
  cronJobsRequestGeneration += 1
  $cronJobs.set(jobs)

  return true
}

export const setCronJobs = (jobs: CronJob[]) => {
  cronJobsRequestGeneration += 1
  $cronJobs.set(jobs)
}

// In-place edit so the cron overlay's mutations (create/edit/delete/pause/…)
// land in the same atom the sidebar renders — no stale list until the next poll.
export const updateCronJobs = (fn: (jobs: CronJob[]) => CronJob[]) => {
  cronJobsRequestGeneration += 1
  $cronJobs.set(fn($cronJobs.get()))
}

// One-shot focus target: clicking "Manage" on a job sets this, then opens the
// cron overlay, which reads it once to select + scroll to that job. Cleared
// after consumption so re-opening cron normally doesn't re-focus a stale job.
export const $cronFocusJobId = atom<null | string>(null)
export const setCronFocusJobId = (id: null | string) => $cronFocusJobId.set(id)
