import type { CronJob } from '@/types/hermes'

// Ledger statuses of an attempt that has not finished (cron/executions.py). The job list only
// reports the newest attempt, and the scheduler rewrites a dead owner's row on its next tick,
// so a claimed/running row is the run the Stop button can reach.
const IN_FLIGHT_STATUSES = new Set(['claimed', 'running'])

type RunFields = Pick<CronJob, 'latest_execution' | 'live_execution'>

// `live_execution` covers a scheduler-fired run hidden behind a newer refused Run-now row.
export function jobRunInFlight(job: RunFields): boolean {
  return IN_FLIGHT_STATUSES.has(job.latest_execution?.status ?? '') || Boolean(job.live_execution)
}

export type StopOutcome = 'cancelled' | 'finished' | 'pending'

/**
 * How a Stop aimed at `executionId` ('' until the cancel endpoint names it) ended, from a job-list
 * snapshot taken in the stop's own backend scope. A removed job's run is stopped by the removal.
 * When a newer row (a refused Run-now) hides the targeted attempt, the accepted cancel is the best
 * evidence left, so it reads as cancelled.
 */
export function stopOutcome(job: RunFields | undefined, executionId: string): StopOutcome {
  if (!job) {
    return 'cancelled'
  }

  // Settle by the targeted attempt, never by "some run is live": a newer run started after the
  // stopped one must not keep this Stop pending (and its button disabled).
  if (executionId) {
    const target = [job.live_execution, job.latest_execution].find(execution => execution?.id === executionId)

    if (!target) {
      return 'cancelled'
    }

    if (IN_FLIGHT_STATUSES.has(target.status ?? '')) {
      return 'pending'
    }

    return target.status === 'cancelled' ? 'cancelled' : 'finished'
  }

  if (jobRunInFlight(job)) {
    return 'pending'
  }

  return job.latest_execution?.status === 'cancelled' ? 'cancelled' : 'finished'
}

// A run created this long before a Run-now click (renderer/backend clock skew) still counts as its run.
const TRIGGER_CLOCK_SLACK_MS = 2000

/**
 * Whether the job has a run to stop. While this window's Run-now is still waiting
 * (`triggerStartedAt`, ms), the run counts as live until the ledger shows an attempt claimed
 * since the click that already finished, so the button does not flicker back to Stop between
 * the run ending and the trigger request returning.
 */
export function jobRunLive(job: RunFields, triggerStartedAt?: number): boolean {
  if (jobRunInFlight(job)) {
    return true
  }

  if (triggerStartedAt === undefined) {
    return false
  }

  const claimedAt = Date.parse(job.latest_execution?.claimed_at ?? '')

  return !(claimedAt >= triggerStartedAt - TRIGGER_CLOCK_SLACK_MS)
}

/** The newest attempt's reason when it was stopped (Stop, `hermes cron cancel`, or removal). */
export function jobLastCancelReason(job: Pick<CronJob, 'latest_execution'>): null | string {
  const execution = job.latest_execution

  return execution?.status === 'cancelled' ? (execution.error ?? '').trim() : null
}

// The editor's missed-run choice. `default` stores nothing, so the job follows
// `cron.catch_up_missed` (recurring) or skips (one-shot), exactly as before the field existed.
export type CatchUpChoice = 'default' | 'run' | 'skip'

export const CATCH_UP_CHOICES: readonly CatchUpChoice[] = ['default', 'run', 'skip']

export function catchUpChoice(value: boolean | null | undefined): CatchUpChoice {
  return value === true ? 'run' : value === false ? 'skip' : 'default'
}

export function catchUpValue(choice: CatchUpChoice): boolean | null {
  return choice === 'run' ? true : choice === 'skip' ? false : null
}
