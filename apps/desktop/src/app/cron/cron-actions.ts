import { type CronJob, getCronJobs, triggerCronJob } from '@/hermes'
import {
  beginCronJobsAction,
  beginCronJobsRequest,
  commitCronJobsRequest,
  type CronJobsRequest,
  isCronJobsRequestCurrent,
  isCronJobsScopeCurrent
} from '@/store/cron'

export interface CronTriggerRefreshResult {
  jobs: CronJob[] | null
  refreshError: unknown | null
  stale: boolean
}

export interface CronMutationRefreshResult<T> extends CronTriggerRefreshResult {
  value: T | null
}

async function refreshForGeneration(
  profile: string,
  request: CronJobsRequest
): Promise<CronTriggerRefreshResult> {
  try {
    const jobs = await getCronJobs(profile)

    if (!commitCronJobsRequest(request, jobs)) {
      return { jobs: null, refreshError: null, stale: true }
    }

    return { jobs, refreshError: null, stale: false }
  } catch (refreshError) {
    if (!isCronJobsRequestCurrent(request)) {
      return { jobs: null, refreshError: null, stale: true }
    }

    return { jobs: null, refreshError, stale: false }
  }
}

export function refreshCronJobs(profile: string): Promise<CronTriggerRefreshResult> {
  return refreshForGeneration(profile, beginCronJobsRequest(profile))
}

export async function mutateAndRefreshCronJobs<T>(
  profile: string,
  mutate: () => Promise<T>
): Promise<CronMutationRefreshResult<T>> {
  const scopeToken = beginCronJobsAction(profile)
  let value: T

  try {
    value = await mutate()
  } catch (mutationError) {
    if (!isCronJobsScopeCurrent(scopeToken)) {
      return { jobs: null, refreshError: null, stale: true, value: null }
    }

    throw mutationError
  }

  if (!isCronJobsScopeCurrent(scopeToken)) {
    return { jobs: null, refreshError: null, stale: true, value: null }
  }

  const refreshed = await refreshCronJobs(profile)

  if (refreshed.stale || !isCronJobsScopeCurrent(scopeToken)) {
    return { jobs: null, refreshError: null, stale: true, value: null }
  }

  return { ...refreshed, value }
}

/**
 * Trigger a job synchronously, then replace the local view from the backend.
 * A completed one-shot may have been deleted, so the trigger response alone is
 * not an authoritative list update. Refresh failure is reported separately:
 * the trigger already succeeded and must not be shown as failed.
 */
export async function triggerAndRefreshCronJobs(
  jobId: string,
  profile: 'all' | string
): Promise<CronTriggerRefreshResult> {
  const { value: _value, ...result } = await mutateAndRefreshCronJobs(profile, () =>
    triggerCronJob(jobId)
  )

  return result
}