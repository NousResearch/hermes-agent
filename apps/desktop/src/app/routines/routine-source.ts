import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'
import { useCallback, useEffect, useState } from 'react'

import type { CronJob, CronOwnerScope } from '@/hermes'
import { useI18n } from '@/i18n'
import { $cronJobs, getCronJobsSnapshotScope, invalidateCronJobsRequests } from '@/store/cron'
import { notifyError } from '@/store/notifications'
import { $sessionListProfileScope, sidebarProfileForScope } from '@/store/profile'

import type { AgentSettingsOwner } from '../agent-settings/data'
import { agentOverviewQueries } from '../agent-settings/overview-queries'
import { cronRequestScope, mutateAndRefreshCronJobs, refreshCronJobs } from '../cron/cron-actions'
import { useRefreshHotkey } from '../hooks/use-refresh-hotkey'

/**
 * Where a Routines surface reads its routines and sends its changes. The rail page and an agent's
 * Settings › Routines render the same rows, detail and editor; they differ only in this: the rail
 * lists every agent's routines from the shared `$cronJobs` snapshot (the sidebar sync already keeps
 * it at every profile in the redesign), an agent's page reads that one owner, on its own machine.
 */
export interface RoutineSource {
  jobs: readonly CronJob[]
  /** Nothing to show yet: the first read has not answered. */
  loading: boolean
  error: unknown
  /** The machine the routines live on (null = the window's own). */
  connectionId: null | string
  /** Edits open the editor here (an agent's Settings › Routines); the rail page links there instead
   *  (board decision, Nov 12: the rail page has quick actions only). */
  editsInPlace: boolean
  /** Owner scope for a call about `job` (its agent, on this source's machine). */
  scopeFor: (job: CronJob) => CronOwnerScope
  /** Owner scope for creating a routine in `profile`. */
  createScope: (profile: string) => CronOwnerScope
  refresh: () => Promise<void>
  /** Run a change, then re-read; null when a connection or profile switch made the answer stale. */
  mutate: <T>(change: () => Promise<T>) => Promise<null | T>
}

/** The rail's Routines page: every agent's routines on this window's machine. */
export function useRailRoutineSource(): RoutineSource {
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
    editsInPlace: false,
    error,
    jobs,
    loading: !answered && jobs.length === 0,
    mutate,
    refresh,
    scopeFor: job => ({ profile: job.profile ?? null })
  }
}

/** Agent › Settings › Routines: the same list, for one agent, read from that agent's own machine. */
export function useAgentRoutineSource(owner: AgentSettingsOwner): RoutineSource {
  const query = useQuery(agentOverviewQueries.routines(owner))
  const { refetch } = query
  const scope: CronOwnerScope = { connectionId: owner.connectionId ?? null, profile: owner.profile }

  const refresh = useCallback(async () => {
    await refetch()
  }, [refetch])

  useRefreshHotkey(refresh)

  const mutate = useCallback(
    async <T,>(change: () => Promise<T>): Promise<null | T> => {
      const value = await change()
      await refetch()

      return value
    },
    [refetch]
  )

  return {
    connectionId: owner.connectionId ?? null,
    createScope: () => scope,
    editsInPlace: true,
    error: query.error,
    jobs: query.data ?? [],
    loading: query.isPending,
    mutate,
    refresh,
    scopeFor: () => scope
  }
}
