import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useRef, useState } from 'react'

import { cancelCronJobRun, type CronJob } from '@/hermes'
import type { Translations } from '@/i18n'
import { getCronJobsSnapshotScope } from '@/store/cron'
import { $cronChangeTick } from '@/store/live-sync'
import { notify, notifyError } from '@/store/notifications'

import { cronRequestScope, mutateAndRefreshCronJobs, refreshCronJobs } from './cron-actions'
import { jobTitle, truncateText } from './job-state'
import { stopOutcome } from './run-control'

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

interface PendingStop {
  /** The execution the cancel targeted; '' until the endpoint names it. */
  executionId: string
  /** Backend scope (connection + profile) the Stop was sent to. */
  scope: string
  title: string
}

/**
 * Stop for a cron job's live run: POST /cancel, then hold "Stopping…" until a job list from the
 * same backend shows that attempt finished. A run that ends on its own first is reported as such,
 * not as cancelled; a Stop left behind by a connection or profile switch is dropped silently.
 */
export function useCronRunStop({ c, jobs, profile, watch }: CronRunStopOptions): CronRunStop {
  const stoppingRef = useRef(new Map<string, PendingStop>())
  const [stoppingJobIds, setStoppingJobIds] = useState<ReadonlySet<string>>(() => new Set())
  const cronChangeTick = useStore($cronChangeTick)

  const publish = useCallback(() => setStoppingJobIds(new Set(stoppingRef.current.keys())), [])

  const settle = useCallback(
    (list: CronJob[] | null) => {
      if (!list) {
        return
      }

      const currentScope = cronRequestScope(profile)
      // A list wiped by a connection switch is no evidence that any run ended.
      const listScope = getCronJobsSnapshotScope()
      let changed = false

      for (const [jobId, { executionId, scope, title }] of stoppingRef.current) {
        if (scope !== currentScope) {
          stoppingRef.current.delete(jobId)
          changed = true

          continue
        }

        const outcome =
          listScope === scope
            ? stopOutcome(
                list.find(candidate => candidate.id === jobId),
                executionId
              )
            : 'pending'

        if (outcome === 'pending') {
          continue
        }

        stoppingRef.current.delete(jobId)
        changed = true
        notify(
          outcome === 'cancelled'
            ? { kind: 'success', title: c.control.stopped, message: title }
            : { kind: 'info', title: c.control.finishedFirst, message: title }
        )
      }

      if (changed) {
        publish()
      }
    },
    [c, profile, publish]
  )

  // Another surface (the poll below, the sidebar, a cron.changed refresh) may publish the
  // settling snapshot; reading the shared atom catches it whichever request wins.
  useEffect(() => settle(jobs), [jobs, settle])

  const active = watch || stoppingJobIds.size > 0

  useEffect(() => {
    if (!active) {
      return
    }

    // One refresh at a time: each request supersedes the one before it, so polls overlapping on a
    // slow backend would keep every reply stale and the page would never leave "Stopping…".
    let timeoutId: number | undefined
    let disposed = false

    const poll = async () => {
      if (document.visibilityState === 'visible') {
        await refreshCronJobs(profile)
      }

      if (!disposed) {
        timeoutId = window.setTimeout(poll, LIVE_RUN_POLL_MS)
      }
    }

    timeoutId = window.setTimeout(poll, LIVE_RUN_POLL_MS)

    return () => {
      disposed = true
      window.clearTimeout(timeoutId)
    }
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
      const scope = cronRequestScope(profile)

      stoppingRef.current.set(job.id, { executionId: '', scope, title })
      publish()

      try {
        const { jobs: refreshed, value } = await mutateAndRefreshCronJobs(profile, () => cancelCronJobRun(job.id))

        if (value && !value.cancelled) {
          stoppingRef.current.delete(job.id)
          publish()
          notify({ kind: 'info', title: c.control.nothingToStop, message: title })

          return
        }

        if (value?.execution_id && stoppingRef.current.get(job.id)?.scope === scope) {
          stoppingRef.current.set(job.id, { executionId: value.execution_id, scope, title })
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
