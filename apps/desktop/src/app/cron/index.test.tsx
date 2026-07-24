// @vitest-environment jsdom
import { QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as HermesApi from '@/hermes'
import type { CronJob, SessionInfo } from '@/hermes'
import { queryClient } from '@/lib/query-client'
import { setCronFocusJobId, setCronJobs } from '@/store/cron'

const getCronJob = vi.fn()
const getCronJobRuns = vi.fn()
const getCronJobs = vi.fn()
const pauseCronJob = vi.fn()
const triggerCronJob = vi.fn()

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof HermesApi>()),
  createCronJob: vi.fn(),
  deleteCronJob: vi.fn(),
  getCronJob: (jobId: string) => getCronJob(jobId),
  getCronJobRuns: (jobId: string) => getCronJobRuns(jobId),
  getCronJobs: (profile?: string) => getCronJobs(profile),
  pauseCronJob: (jobId: string) => pauseCronJob(jobId),
  resumeCronJob: vi.fn(),
  triggerCronJob: (jobId: string) => triggerCronJob(jobId),
  updateCronJob: vi.fn()
}))

vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: vi.fn()
}))

const job: CronJob = {
  deliver: 'local',
  enabled: true,
  id: 'job-1',
  name: 'Status report',
  next_run_at: '2026-07-24T12:00:00Z',
  prompt: 'Prepare a status report',
  schedule: { expr: '*/5 * * * *', kind: 'cron' },
  schedule_display: 'Every 5 minutes',
  state: 'scheduled'
}

const scriptJob: CronJob = {
  ...job,
  no_agent: true,
  script: '/tmp/status.py'
}

const otherJob: CronJob = {
  ...job,
  id: 'job-2',
  name: 'Weekly release notes'
}

const run: SessionInfo = {
  id: 'cron_job-1_20260724_120001',
  last_active: 1_774_612_801,
  started_at: 1_774_612_801,
  title: 'Status report run'
} as SessionInfo

function deferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void

  const promise = new Promise<T>(done => {
    resolve = done
  })

  return { promise, resolve }
}

async function renderCron() {
  const { CronView } = await import('./index')
  let result!: ReturnType<typeof render>

  await act(async () => {
    result = render(
      <QueryClientProvider client={queryClient}>
        <CronView onClose={vi.fn()} />
      </QueryClientProvider>
    )
  })

  return result
}

async function renderLoadedCron() {
  const result = await renderCron()

  await screen.findByText('No runs yet')

  return result
}

function rowButton(name: string) {
  const row = screen.getByText(name).closest('button')

  if (!row) {
    throw new Error(`${name} row is not a button`)
  }

  return row
}

beforeAll(() => {
  HTMLElement.prototype.scrollIntoView ??= () => undefined
  vi.stubGlobal('CSS', { escape: (value: string) => value })
})

beforeEach(() => {
  setCronJobs([])
  setCronFocusJobId(null)
  getCronJob.mockResolvedValue(job)
  getCronJobs.mockResolvedValue([job])
  getCronJobRuns.mockResolvedValue([])
  pauseCronJob.mockResolvedValue({ ...job, state: 'paused' })
})

afterEach(() => {
  vi.clearAllTimers()
  vi.useRealTimers()
  cleanup()
  queryClient.clear()
  setCronJobs([])
  setCronFocusJobId(null)
  vi.clearAllMocks()
})

describe('CronView trigger feedback', () => {
  it('blocks same-tick duplicate triggers and paints pending feedback immediately', async () => {
    const trigger = deferred<CronJob>()

    triggerCronJob.mockReturnValue(trigger.promise)
    const { container } = await renderCron()
    const triggerButton = await screen.findByRole('button', { name: 'Trigger now' })

    await act(async () => {
      triggerButton.click()
      triggerButton.click()
      await Promise.resolve()
    })

    expect(triggerCronJob).toHaveBeenCalledOnce()
    expect((triggerButton as HTMLButtonElement).disabled).toBe(true)
    expect(triggerButton.getAttribute('aria-busy')).toBe('true')
    expect(triggerButton.querySelector('.codicon-loading')).toBeTruthy()
    expect(container.querySelector('[data-slot="cron-run-pending"]')).toBeTruthy()

    await act(async () => trigger.resolve(job))
  })

  it('keeps pause available after the trigger request is queued', async () => {
    const nextRuns = deferred<SessionInfo[]>()

    triggerCronJob.mockResolvedValue(job)
    await renderLoadedCron()
    getCronJobRuns.mockImplementation(() => nextRuns.promise)

    await act(async () => screen.getByRole('button', { name: 'Trigger now' }).click())

    expect((screen.getByRole('button', { name: 'Trigger now' }) as HTMLButtonElement).disabled).toBe(true)
    expect((screen.getByRole('button', { name: 'Pause' }) as HTMLButtonElement).disabled).toBe(false)

    await act(async () => nextRuns.resolve([]))
  })

  it('clears queued feedback when pausing cancels the pending run', async () => {
    const nextRuns = deferred<SessionInfo[]>()

    triggerCronJob.mockResolvedValue(job)
    const { container } = await renderLoadedCron()
    getCronJobRuns.mockImplementation(() => nextRuns.promise)

    await act(async () => screen.getByRole('button', { name: 'Trigger now' }).click())
    expect(container.querySelector('[data-slot="cron-run-pending"]')).toBeTruthy()

    await act(async () => screen.getByRole('button', { name: 'Pause' }).click())

    expect(pauseCronJob).toHaveBeenCalledWith(job.id)
    expect(container.querySelector('[data-slot="cron-run-pending"]')).toBeFalsy()
    expect((screen.getByRole('button', { name: 'Trigger now' }) as HTMLButtonElement).disabled).toBe(false)

    await act(async () => nextRuns.resolve([]))
  })

  it('does not let a canceled job probe overwrite a successful pause', async () => {
    const jobProbe = deferred<CronJob>()
    const pendingScriptJob: CronJob = { ...scriptJob, last_run_at: '2026-07-24T12:00:00Z' }
    const pausedJob: CronJob = { ...pendingScriptJob, enabled: false, state: 'paused' }
    const completedJob: CronJob = { ...pendingScriptJob, last_run_at: '2026-07-24T12:00:01Z' }

    getCronJobs.mockResolvedValue([pendingScriptJob])
    getCronJob.mockImplementation(() => jobProbe.promise)
    triggerCronJob.mockResolvedValue(pendingScriptJob)
    pauseCronJob.mockResolvedValue(pausedJob)
    await renderLoadedCron()

    await act(async () => screen.getByRole('button', { name: 'Trigger now' }).click())
    await waitFor(() => expect(getCronJob).toHaveBeenCalledWith(pendingScriptJob.id))
    await act(async () => screen.getByRole('button', { name: 'Pause' }).click())
    await screen.findByRole('button', { name: 'Resume' })

    await act(async () => jobProbe.resolve(completedJob))

    expect(screen.getByRole('button', { name: 'Resume' })).toBeTruthy()
  })

  it('replaces the queued run with the authoritative run and unlocks the action', async () => {
    const nextRuns = deferred<SessionInfo[]>()

    triggerCronJob.mockResolvedValue(job)
    const { container } = await renderLoadedCron()
    getCronJobRuns.mockImplementation(() => nextRuns.promise)
    const triggerButton = screen.getByRole('button', { name: 'Trigger now' })

    await act(async () => triggerButton.click())

    expect(container.querySelector('[data-slot="cron-run-pending"]')).toBeTruthy()
    expect((triggerButton as HTMLButtonElement).disabled).toBe(true)

    const observedAt = Date.now() / 1000

    await act(async () => nextRuns.resolve([{ ...run, last_active: observedAt, started_at: observedAt }]))

    await screen.findByText('Status report run')
    await waitFor(() => expect(container.querySelector('[data-slot="cron-run-pending"]')).toBeNull())
    expect((triggerButton as HTMLButtonElement).disabled).toBe(false)
  })

  it('settles a runless script job from its completed job state', async () => {
    const completedJob: CronJob = {
      ...scriptJob,
      last_run_at: new Date(Date.now() + 1000).toISOString()
    }

    getCronJobs.mockResolvedValue([scriptJob])
    getCronJob.mockResolvedValue(completedJob)
    triggerCronJob.mockResolvedValue(scriptJob)
    const { container } = await renderLoadedCron()

    await act(async () => screen.getByRole('button', { name: 'Trigger now' }).click())

    await waitFor(() => expect(getCronJob).toHaveBeenCalledWith(scriptJob.id))
    expect(container.querySelector('[data-slot="cron-run-pending"]')).toBeFalsy()
    expect((screen.getByRole('button', { name: 'Trigger now' }) as HTMLButtonElement).disabled).toBe(false)
  })

  it('settles from a changed job timestamp instead of the backend clock', async () => {
    const previousRunAt = '2099-01-01T00:00:00Z'
    const skewedJob: CronJob = { ...job, last_run_at: previousRunAt }

    getCronJobs.mockResolvedValue([skewedJob])
    getCronJob.mockResolvedValue(skewedJob)
    triggerCronJob.mockResolvedValue(skewedJob)
    const { container } = await renderLoadedCron()
    vi.useFakeTimers()

    await act(async () => screen.getByRole('button', { name: 'Trigger now' }).click())
    await act(async () => Promise.resolve())

    expect(container.querySelector('[data-slot="cron-run-pending"]')).toBeTruthy()

    getCronJob.mockResolvedValue({ ...skewedJob, last_run_at: '2099-01-01T00:00:01Z' })
    await act(async () => vi.advanceTimersByTimeAsync(1000))

    expect(container.querySelector('[data-slot="cron-run-pending"]')).toBeFalsy()
  })

  it('uses the completed job timestamp as the baseline for the next runless trigger', async () => {
    const firstTrigger = deferred<CronJob>()
    const firstCompletion = deferred<CronJob>()
    const secondTrigger = deferred<CronJob>()
    const pendingScriptJob: CronJob = { ...scriptJob, last_run_at: '2026-07-24T12:00:00Z' }
    const completedJob: CronJob = { ...pendingScriptJob, last_run_at: '2026-07-24T12:00:01Z' }

    getCronJobs.mockResolvedValue([pendingScriptJob])
    getCronJob.mockImplementationOnce(() => firstCompletion.promise)
    triggerCronJob.mockImplementationOnce(() => firstTrigger.promise)
    const { container } = await renderLoadedCron()

    await act(async () => screen.getByRole('button', { name: 'Trigger now' }).click())
    await act(async () => firstCompletion.resolve(completedJob))
    await waitFor(() => expect(container.querySelector('[data-slot="cron-run-pending"]')).toBeFalsy())
    await act(async () => firstTrigger.resolve(pendingScriptJob))
    await waitFor(() => {
      expect((screen.getByRole('button', { name: 'Trigger now' }) as HTMLButtonElement).disabled).toBe(false)
    })

    getCronJob.mockResolvedValue(completedJob)
    triggerCronJob.mockImplementationOnce(() => secondTrigger.promise)
    await act(async () => screen.getByRole('button', { name: 'Trigger now' }).click())
    await act(async () => Promise.resolve())

    expect(container.querySelector('[data-slot="cron-run-pending"]')).toBeTruthy()

    await act(async () => secondTrigger.resolve(completedJob))
  })

  it('does not stack queued run polls while a request is still in flight', async () => {
    const nextRuns = deferred<SessionInfo[]>()

    triggerCronJob.mockResolvedValue(job)
    await renderLoadedCron()
    getCronJobRuns.mockClear()
    getCronJobRuns.mockImplementation(() => nextRuns.promise)
    vi.useFakeTimers()

    await act(async () => screen.getByRole('button', { name: 'Trigger now' }).click())

    expect(getCronJobRuns).toHaveBeenCalledOnce()

    await act(async () => vi.advanceTimersByTimeAsync(1000))

    expect(getCronJobRuns).toHaveBeenCalledOnce()
    await act(async () => nextRuns.resolve([]))
  })

  it('serializes run-history loads across a pending-state effect restart', async () => {
    const initialRuns = deferred<SessionInfo[]>()

    getCronJobRuns.mockImplementation(() => initialRuns.promise)
    triggerCronJob.mockResolvedValue(job)
    await renderCron()
    const triggerButton = await screen.findByRole('button', { name: 'Trigger now' })
    await waitFor(() => expect(getCronJobRuns).toHaveBeenCalledOnce())

    await act(async () => {
      triggerButton.click()
      await Promise.resolve()
      await Promise.resolve()
    })

    expect(getCronJobRuns).toHaveBeenCalledOnce()

    await act(async () => initialRuns.resolve([]))
    await waitFor(() => expect(getCronJobRuns).toHaveBeenCalledTimes(2))
  })

  it('unlocks the action when a queued run does not appear before the timeout', async () => {
    const nextRuns = deferred<SessionInfo[]>()

    triggerCronJob.mockResolvedValue(job)
    const { container } = await renderLoadedCron()
    getCronJobRuns.mockImplementation(() => nextRuns.promise)
    vi.useFakeTimers()

    const triggerButton = screen.getByRole('button', { name: 'Trigger now' })

    await act(async () => triggerButton.click())

    expect((triggerButton as HTMLButtonElement).disabled).toBe(true)
    expect(container.querySelector('[data-slot="cron-run-pending"]')).toBeTruthy()

    await act(async () => vi.advanceTimersByTimeAsync(90_000))

    expect((triggerButton as HTMLButtonElement).disabled).toBe(false)
    expect(container.querySelector('[data-slot="cron-run-pending"]')).toBeNull()
  })

  it('keeps the original timeout deadline after selecting another job', async () => {
    getCronJobs.mockResolvedValue([job, otherJob])
    triggerCronJob.mockResolvedValue(job)
    const { container } = await renderLoadedCron()
    vi.useFakeTimers()

    await act(async () => screen.getByRole('button', { name: 'Trigger now' }).click())
    await act(async () => vi.advanceTimersByTimeAsync(60_000))
    await act(async () => rowButton('Weekly release notes').click())
    expect(screen.getByRole('heading', { name: 'Weekly release notes' })).toBeTruthy()
    await act(async () => vi.advanceTimersByTimeAsync(31_000))
    await act(async () => rowButton('Status report').click())
    await act(async () => vi.advanceTimersByTimeAsync(0))

    expect((screen.getByRole('button', { name: 'Trigger now' }) as HTMLButtonElement).disabled).toBe(false)
    expect(container.querySelector('[data-slot="cron-run-pending"]')).toBeNull()
  })

  it('removes pending feedback and unlocks the action when triggering fails', async () => {
    triggerCronJob.mockRejectedValue(new Error('scheduler unavailable'))
    const { container } = await renderLoadedCron()
    const triggerButton = screen.getByRole('button', { name: 'Trigger now' })

    await act(async () => triggerButton.click())

    await waitFor(() => expect(container.querySelector('[data-slot="cron-run-pending"]')).toBeNull())
    expect((triggerButton as HTMLButtonElement).disabled).toBe(false)
    expect(triggerButton.querySelector('.codicon-zap')).toBeTruthy()
  })
})
