// @vitest-environment jsdom
import { QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, render, renderHook, screen } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as HermesApi from '@/hermes'
import type { CronJob } from '@/hermes'
import type { Translations } from '@/i18n'
import { queryClient } from '@/lib/query-client'
import { $cronJobs, invalidateCronJobsRequests, setCronFocusJobId, setCronJobs } from '@/store/cron'
import { setChangeEventsAvailable } from '@/store/live-sync'
import { notify } from '@/store/notifications'
import type { CronCancelResult } from '@/types/hermes'

import { useCronRunStop } from './use-cron-run-stop'

const cancelCronJobRun = vi.fn()
const getCronJobRuns = vi.fn()
const getCronJobs = vi.fn()

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof HermesApi>()),
  cancelCronJobRun: (jobId: string) => cancelCronJobRun(jobId),
  getCronJobRuns: (jobId: string) => getCronJobRuns(jobId),
  getCronJobs: (profile?: string) => getCronJobs(profile)
}))

vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: vi.fn()
}))

const c = {
  control: { failedStop: 'failed', finishedFirst: 'finished first', nothingToStop: 'nothing', stopped: 'Run cancelled' }
} as unknown as Translations['cron']

const running: CronJob = {
  deliver: 'local',
  enabled: true,
  id: 'job-1',
  latest_execution: { claimed_at: '2026-10-25T21:00:00Z', id: 'exec-1', status: 'running' },
  name: 'Nightly digest',
  prompt: 'Summarize',
  schedule: { expr: '0 3 * * *', kind: 'cron' },
  schedule_display: 'Every day at 03:00',
  state: 'scheduled'
} as CronJob

beforeAll(() => {
  HTMLElement.prototype.scrollIntoView ??= () => undefined
  vi.stubGlobal('CSS', { escape: (value: string) => value })
})

beforeEach(() => {
  setCronFocusJobId(null)
  setChangeEventsAvailable(false)
  getCronJobRuns.mockResolvedValue([])
})

afterEach(() => {
  vi.useRealTimers()
  cleanup()
  queryClient.clear()
  invalidateCronJobsRequests()
  setCronJobs([])
  vi.clearAllMocks()
})

describe('cron job header while a run is live', () => {
  it('offers Resume for a paused job whose last run is still going', async () => {
    const pausedRunning = { ...running, enabled: false, state: 'paused' } as CronJob
    setCronJobs([pausedRunning])
    getCronJobs.mockResolvedValue([pausedRunning])
    const { CronView } = await import('./index')

    await act(async () => {
      render(
        <QueryClientProvider client={queryClient}>
          <CronView onClose={vi.fn()} />
        </QueryClientProvider>
      )
    })

    // The click resumes the schedule (handlePauseResume reads the saved state), so the label must say so.
    expect(await screen.findByRole('button', { name: 'Resume' })).toBeTruthy()
    expect(screen.queryByRole('button', { name: 'Pause' })).toBeNull()
  })
})

describe('useCronRunStop', () => {
  it('lets a slow job-list refresh land instead of superseding it every poll', async () => {
    vi.useFakeTimers()
    const stopped = { ...running, latest_execution: { ...running.latest_execution!, status: 'cancelled' } } as CronJob
    let inFlight = 0
    let maxInFlight = 0
    getCronJobs.mockImplementation(
      () =>
        new Promise(resolve => {
          inFlight += 1
          maxInFlight = Math.max(maxInFlight, inFlight)
          setTimeout(() => {
            inFlight -= 1
            resolve([stopped])
          }, 4000)
        })
    )

    renderHook(() => useCronRunStop({ c, jobs: [running], profile: 'default', watch: true }))
    await act(() => vi.advanceTimersByTimeAsync(12_000))

    expect(maxInFlight).toBe(1)
    expect($cronJobs.get()).toEqual([stopped])
  })

  it('does not announce a cancel when a connection switch wipes the job list', async () => {
    cancelCronJobRun.mockReturnValue(new Promise<CronCancelResult>(() => undefined))

    const { rerender, result } = renderHook(
      ({ jobs }) => useCronRunStop({ c, jobs, profile: 'default', watch: false }),
      {
        initialProps: { jobs: [running] }
      }
    )

    act(() => void result.current.stop(running))
    expect(result.current.stoppingJobIds.has('job-1')).toBe(true)

    // gateway-switch.ts clears the cache while the cancel request is still pending.
    act(() => setCronJobs([]))
    rerender({ jobs: [] })

    expect(notify).not.toHaveBeenCalled()
    expect(result.current.stoppingJobIds.has('job-1')).toBe(true)
  })
})
