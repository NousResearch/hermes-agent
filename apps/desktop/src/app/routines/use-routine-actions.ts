import { createCronTriggerController } from '@hermes/shared'
import { useEffect, useState } from 'react'

import { cancelCronJobRun, type CronJob, deleteCronJob, pauseCronJob, resumeCronJob, triggerCronJob } from '@/hermes'
import { useI18n } from '@/i18n'
import { notify, notifyError } from '@/store/notifications'

import { jobState, truncateText } from '../cron/job-state'
import { jobRunInFlight } from '../cron/run-control'

import { routineTitle } from './routine-model'
import type { RoutineSource } from './routine-source'
import { routineKey } from './routines-store'

// A cancel lands at the run's next cancel check; poll this fast while a run is live so its row and
// buttons settle without a manual refresh (the classic page's cadence).
const LIVE_RUN_POLL_MS = 1500

export interface RoutineActions {
  /** Run now, and Retry on a failed routine: one action, the backend's synchronous trigger. */
  runNow: (job: CronJob) => Promise<void>
  pauseResume: (job: CronJob) => Promise<void>
  stop: (job: CronJob) => Promise<void>
  /** Throws on failure, so the confirm dialog reports it inline and stays open. */
  remove: (job: CronJob) => Promise<void>
  /** A run is in flight (ledger) or this window's Run now is still waiting on it. */
  isLive: (job: CronJob) => boolean
  isBusy: (job: CronJob) => boolean
  isStopping: (job: CronJob) => boolean
}

/** Keys of the routines an action is in flight for, as one immutable set per change. */
function useKeySet() {
  const [keys, setKeys] = useState<ReadonlySet<string>>(() => new Set())

  const toggle = (key: string, on: boolean) =>
    setKeys(current => {
      if (current.has(key) === on) {
        return current
      }

      const next = new Set(current)

      if (on) {
        next.add(key)
      } else {
        next.delete(key)
      }

      return next
    })

  return [keys, toggle] as const
}

const title = (job: CronJob) => truncateText(routineTitle(job), 60)

export function useRoutineActions(source: RoutineSource): RoutineActions {
  const { t } = useI18n()
  const c = t.cron
  const [triggering, setTriggering] = useKeySet()
  const [busy, setBusy] = useKeySet()
  const [stopping, setStopping] = useKeySet()
  // One guard per mounted surface: a second click on a routine already starting is ignored.
  const [controller] = useState(() => createCronTriggerController((key, on) => setTriggering(key, on)))
  const { mutate, refresh, scopeFor } = source

  async function runNow(job: CronJob) {
    try {
      const run = await controller.run(
        routineKey(job),
        () => mutate(() => triggerCronJob(job.id, scopeFor(job))),
        () => notify({ kind: 'info', title: c.triggerNow, message: title(job) })
      )

      if (run.started && run.value) {
        notify({ kind: 'success', title: c.triggered, message: title(job) })
      }
    } catch (err) {
      notifyError(err, c.failedTrigger)
    }
  }

  async function pauseResume(job: CronJob) {
    const paused = jobState(job) === 'paused'
    const key = routineKey(job)

    setBusy(key, true)

    try {
      const scope = scopeFor(job)
      const done = await mutate(() => (paused ? resumeCronJob(job.id, scope) : pauseCronJob(job.id, scope)))

      if (done) {
        notify({ kind: 'success', title: paused ? c.resumed : c.paused, message: title(job) })
      }
    } catch (err) {
      notifyError(err, c.failedUpdate)
    } finally {
      setBusy(key, false)
    }
  }

  async function stop(job: CronJob) {
    const key = routineKey(job)

    setStopping(key, true)

    try {
      const result = await mutate(() => cancelCronJobRun(job.id, scopeFor(job)))

      if (result) {
        notify(
          result.cancelled
            ? { kind: 'success', title: c.control.stopped, message: title(job) }
            : { kind: 'info', title: c.control.nothingToStop, message: title(job) }
        )
      }
    } catch (err) {
      notifyError(err, c.control.failedStop)
    } finally {
      setStopping(key, false)
    }
  }

  async function remove(job: CronJob) {
    const done = await mutate(() => deleteCronJob(job.id, scopeFor(job)))

    if (done) {
      notify({ kind: 'success', title: c.deleted, message: title(job) })
    }
  }

  const isLive = (job: CronJob) => jobRunInFlight(job) || triggering.has(routineKey(job))
  const anyLive = source.jobs.some(isLive) || stopping.size > 0

  // While anything runs here, re-read until it settles; the backend's own events refresh the rest.
  useEffect(() => {
    if (!anyLive) {
      return
    }

    const timer = window.setInterval(() => {
      if (document.visibilityState === 'visible') {
        void refresh()
      }
    }, LIVE_RUN_POLL_MS)

    return () => window.clearInterval(timer)
  }, [anyLive, refresh])

  return {
    isBusy: job => busy.has(routineKey(job)) || triggering.has(routineKey(job)),
    isLive,
    isStopping: job => stopping.has(routineKey(job)),
    pauseResume,
    remove,
    runNow,
    stop
  }
}
