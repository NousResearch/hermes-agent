import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useMemo, useState } from 'react'

import type { CronJob, CronOwnerScope } from '@/hermes'
import { useI18n } from '@/i18n'
import { $cronJobs, getCronJobsSnapshotScope, invalidateCronJobsRequests } from '@/store/cron'
import { notifyError } from '@/store/notifications'
import { $sessionListProfileScope, sidebarProfileForScope } from '@/store/profile'

import { cronRequestScope, mutateAndRefreshCronJobs, refreshCronJobs } from '../cron/cron-actions'
import { useRefreshHotkey } from '../hooks/use-refresh-hotkey'

import { $heldRoutine } from './routine-held-delete'

/**
 * Where the Routines page reads its routines and sends its changes: every agent's routines on this
 * window's machine, from the shared `$cronJobs` snapshot (the sidebar sync already keeps it at every
 * profile in the redesign). A routine whose delete is being held (routine-held-delete.ts) is not listed.
 */
export interface RoutineSource {
  jobs: readonly CronJob[]
  /** Nothing to show yet: the first read has not answered. */
  loading: boolean
  error: unknown
  /** The machine the routines live on (null = the window's own). */
  connectionId: null | string
  /** Owner scope for a call about `job` (its agent, on this source's machine). */
  scopeFor: (job: CronJob) => CronOwnerScope
  /** Owner scope for creating a routine in `profile`. */
  createScope: (profile: string) => CronOwnerScope
  refresh: () => Promise<void>
  /** Run a change, then re-read; null when a connection or profile switch made the answer stale. */
  mutate: <T>(change: () => Promise<T>) => Promise<null | T>
}

/** A source without the routine whose delete is being held. */
function useHeldOut(source: RoutineSource): RoutineSource {
  const held = useStore($heldRoutine)
  const { connectionId, jobs: all } = source

  const jobs = useMemo(
    () =>
      held && held.scope.connectionId === connectionId
        ? all.filter(job => !(job.id === held.id && (job.profile ?? '') === held.scope.profile))
        : all,
    [all, connectionId, held]
  )

  return jobs === all ? source : { ...source, jobs }
}

/** The Routines page's routines: every agent's on this window's machine. */
export function useRailRoutineSource(): RoutineSource {
  return useHeldOut(useWindowRoutineSource())
}

/** Every agent's routines on this window's machine. */
function useWindowRoutineSource(): RoutineSource {
  const { t } = useI18n()
  const jobs = useStore($cronJobs)
  const listProfile = sidebarProfileForScope(useStore($sessionListProfileScope))
  // A snapshot taken for another scope (or wiped by a switch) is no answer for this one.
  const [answered, setAnswered] = useState(() => getCronJobsSnapshotScope() === cronRequestScope(listProfile))
  const [error, setError] = useState<unknown>(null)

  const refresh = useCallback(async () => {
    const { refreshError, stale } = await refreshCronJobs(listProfile)

    if (stale) {
      return
    }

    setError(refreshError)
    setAnswered(true)

    if (refreshError) {
      notifyError(refreshError, t.cron.failedLoad)
    }
  }, [listProfile, t])

  useRefreshHotkey(refresh)

  useEffect(() => {
    void refresh()

    // Fence the previous scope's request, and every pending completion on unmount (as CronView does).
    return () => invalidateCronJobsRequests()
  }, [refresh])

  const mutate = useCallback(
    async <T,>(change: () => Promise<T>): Promise<null | T> => {
      const { refreshError, stale, value } = await mutateAndRefreshCronJobs(listProfile, change)

      if (refreshError) {
        notifyError(refreshError, t.cron.failedLoad)
      }

      return stale ? null : value
    },
    [listProfile, t]
  )

  return {
    connectionId: null,
    createScope: profile => ({ profile }),
    error,
    jobs,
    loading: !answered && jobs.length === 0,
    mutate,
    refresh,
    scopeFor: job => ({ profile: job.profile ?? null })
  }
}
