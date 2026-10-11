import { atom } from 'nanostores'

import type { CronJob, CronJobCarriedFields } from '@/types/hermes'

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

// The cron editor's unsaved form, as the user left it. Every field is the
// dialog's own state (modelChoice is its opaque provider/model value).
export interface CronEditorDraftValues {
  /** Settings a copied job or a customized recipe keeps without showing them. */
  carried: CronJobCarriedFields
  deliver: string
  modelChoice: string
  name: string
  prompt: string
  schedule: string
  schedulePreset: string
}

export interface CronEditorDraft {
  /** The job being edited; null for a new job. */
  jobId: null | string
  /** The cron profile scope the draft belongs to; it never reopens in another. */
  profile: string
  values: CronEditorDraftValues
}

// Parked when "Test in new chat" or "Connect a platform" takes the user out of
// the Cron overlay mid-edit, so the round trip doesn't cost them the form. The
// overlay takes it once on its next mount; in-memory only, one per window.
const $cronEditorDraft = atom<CronEditorDraft | null>(null)

export const parkCronEditorDraft = (draft: CronEditorDraft) => $cronEditorDraft.set(draft)

export function takeCronEditorDraft(profile: string): CronEditorDraft | null {
  const draft = $cronEditorDraft.get()

  $cronEditorDraft.set(null)

  return draft?.profile === profile ? draft : null
}

// Shell-owned one-shot intent for stores without router context. Do not set a
// focus id here: the cron overlay's first fetch may not have loaded that row.
export const $cronReviewRequest = atom(0)
export const requestCronReview = () => $cronReviewRequest.set($cronReviewRequest.get() + 1)
