import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useRef, useState } from 'react'

import { cancelCronJobRun, type CronJob } from '@/hermes'
import type { Translations } from '@/i18n'
import { $cronChangeTick } from '@/store/live-sync'
import { notify, notifyError } from '@/store/notifications'

import { mutateAndRefreshCronJobs, refreshCronJobs } from './cron-actions'
import { jobTitle, truncateText } from './job-state'
import { jobRunInFlight } from './run-control'

// A cancel lands at the run's next cancel check (about a second); poll the job list that
// fast while a run is live or stopping so the button settles without a manual refresh.
const LIVE_RUN_POLL_MS = 1500

interface CronRunStopOptions {
  c: Translations['cron']
  jobs: CronJob[]
  profile: string
  /** The selected job has a run in flight (ledger) or a Run-now still waiting. */
  watch: boolean
}

export interface CronRunStop {
  stop: (job: CronJob) => Promise<void>
  stoppingJobIds: ReadonlySet<string>
}

/**
 * Stop for a cron job's live run: POST /cancel, then hold "Stopping…" until the job list
 * shows that attempt finished. A run that ends on its own first is reported as such, not
 * as cancelled.
 */
export function useCronRunStop({ c, jobs, profile, watch }: CronRunStopOptions): CronRunStop {
  // jobId → the execution the cancel targeted ('' until the endpoint names it) + its title.
  const stoppingRef = useRef(new Map<string, { executionId: string; title: string }>())
  const [stoppingJobIds, setStoppingJobIds] = useState<ReadonlySet<string>>(() => new Set())
  const cronChangeTick = useStore($cronChangeTick)

  const publish = useCallback(() => setStoppingJobIds(new Set(stoppingRef.current.keys())), [])

  const settle = useCallback(
    (list: CronJob[] | null) => {
      if (!list) {
        return
      }

      let changed = false

      for (const [jobId, { executionId, title }] of stoppingRef.current) {
        const job = list.find(candidate => candidate.id === jobId)

        if (job && jobRunInFlight(job)) {
          continue
        }

        stoppingRef.current.delete(jobId)
        changed = true

        const execution = job?.latest_execution
        // A removed job's run is stopped by the removal itself.
        const cancelled = !job || (execution?.status === 'cancelled' && (!executionId || execution.id === executionId))

        notify(
          cancelled
            ? { kind: 'success', title: c.control.stopped, message: title }
            : { kind: 'info', title: c.control.finishedFirst, message: title }
        )
      }

      if (changed) {
        publish()
      }
    },
    [c, publish]
  )

  // Another surface (the poll below, the sidebar, a cron.changed refresh) may publish the
  // settling snapshot; reading the shared atom catches it whichever request wins.
  useEffect(() => settle(jobs), [jobs, settle])

  const active = watch || stoppingJobIds.size > 0

  useEffect(() => {
    if (!active) {
      return
    }

    const intervalId = window.setInterval(() => {
      if (document.visibilityState === 'visible') {
        void refreshCronJobs(profile)
      }
    }, LIVE_RUN_POLL_MS)

    return () => window.clearInterval(intervalId)
  }, [active, profile])

  // A scheduler fire rewrites jobs.json (cron.changed): refresh so a run the user did not
  // start here still shows its Stop button.
  useEffect(() => {
    if (cronChangeTick > 0) {
      void refreshCronJobs(profile)
    }
  }, [cronChangeTick, profile])

  const stop = useCallback(
    async (job: CronJob) => {
      if (stoppingRef.current.has(job.id)) {
        return
      }

      const title = truncateText(jobTitle(job), 60)

      stoppingRef.current.set(job.id, { executionId: '', title })
      publish()

      try {
        const { jobs: refreshed, value } = await mutateAndRefreshCronJobs(profile, () => cancelCronJobRun(job.id))

        if (value && !value.cancelled) {
          stoppingRef.current.delete(job.id)
          publish()
          notify({ kind: 'info', title: c.control.nothingToStop, message: title })

          return
        }

        if (value?.execution_id && stoppingRef.current.has(job.id)) {
          stoppingRef.current.set(job.id, { executionId: value.execution_id, title })
        }

        settle(refreshed)
      } catch (err) {
        stoppingRef.current.delete(job.id)
        publish()
        notifyError(err, c.control.failedStop)
      }
    },
    [c, profile, publish, settle]
  )

  return { stop, stoppingJobIds }
}
