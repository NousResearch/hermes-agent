// @vitest-environment jsdom
import { QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as HermesApi from '@/hermes'
import type { CronJob, SessionInfo } from '@/hermes'
import { queryClient } from '@/lib/query-client'
import { setCronFocusJobId, setCronJobs } from '@/store/cron'

const getCronJobRuns = vi.fn()
const getCronJobs = vi.fn()
const triggerCronJob = vi.fn()

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof HermesApi>()),
  createCronJob: vi.fn(),
  deleteCronJob: vi.fn(),
  getCronJobRuns: (jobId: string) => getCronJobRuns(jobId),
  getCronJobs: (profile?: string) => getCronJobs(profile),
  pauseCronJob: vi.fn(),
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

beforeAll(() => {
  HTMLElement.prototype.scrollIntoView ??= () => undefined
  vi.stubGlobal('CSS', { escape: (value: string) => value })
})

beforeEach(() => {
  setCronJobs([])
  setCronFocusJobId(null)
  getCronJobs.mockResolvedValue([job])
  getCronJobRuns.mockResolvedValue([])
})

afterEach(() => {
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

  it('replaces the queued run with the authoritative run and unlocks the action', async () => {
    const nextRuns = deferred<SessionInfo[]>()

    triggerCronJob.mockResolvedValue(job)
    const { container } = await renderCron()

    await screen.findByText('No runs yet')
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

  it('does not stack queued run polls while a request is still in flight', async () => {
    const nextRuns = deferred<SessionInfo[]>()

    triggerCronJob.mockResolvedValue(job)
    await renderCron()
    await screen.findByText('No runs yet')
    getCronJobRuns.mockClear()
    getCronJobRuns.mockImplementation(() => nextRuns.promise)
    vi.useFakeTimers()

    try {
      await act(async () => screen.getByRole('button', { name: 'Trigger now' }).click())

      expect(getCronJobRuns).toHaveBeenCalledOnce()

      await act(async () => vi.advanceTimersByTimeAsync(1000))

      expect(getCronJobRuns).toHaveBeenCalledOnce()
      await act(async () => nextRuns.resolve([]))
    } finally {
      vi.clearAllTimers()
      vi.useRealTimers()
    }
  })

  it('unlocks the action when a queued run does not appear before the timeout', async () => {
    const nextRuns = deferred<SessionInfo[]>()

    triggerCronJob.mockResolvedValue(job)
    const { container } = await renderCron()
    await screen.findByText('No runs yet')
    getCronJobRuns.mockImplementation(() => nextRuns.promise)
    vi.useFakeTimers()

    try {
      const triggerButton = screen.getByRole('button', { name: 'Trigger now' })

      await act(async () => triggerButton.click())

      expect((triggerButton as HTMLButtonElement).disabled).toBe(true)
      expect(container.querySelector('[data-slot="cron-run-pending"]')).toBeTruthy()

      await act(async () => vi.advanceTimersByTimeAsync(90_000))

      expect((triggerButton as HTMLButtonElement).disabled).toBe(false)
      expect(container.querySelector('[data-slot="cron-run-pending"]')).toBeNull()
    } finally {
      vi.clearAllTimers()
      vi.useRealTimers()
    }
  })

  it('keeps the original timeout deadline after selecting another job', async () => {
    getCronJobs.mockResolvedValue([job, otherJob])
    triggerCronJob.mockResolvedValue(job)
    const { container } = await renderCron()

    await screen.findByText('No runs yet')
    vi.useFakeTimers()

    try {
      await act(async () => screen.getByRole('button', { name: 'Trigger now' }).click())
      await act(async () => vi.advanceTimersByTimeAsync(60_000))
      const releaseNotesRow = screen.getByText('Weekly release notes').closest('button')

      if (!releaseNotesRow) {
        throw new Error('Weekly release notes row is not a button')
      }

      await act(async () => releaseNotesRow.click())
      expect(screen.getByRole('heading', { name: 'Weekly release notes' })).toBeTruthy()
      await act(async () => vi.advanceTimersByTimeAsync(31_000))
      const statusReportRow = screen.getByText('Status report').closest('button')

      if (!statusReportRow) {
        throw new Error('Status report row is not a button')
      }

      await act(async () => statusReportRow.click())
      await act(async () => vi.advanceTimersByTimeAsync(0))

      expect((screen.getByRole('button', { name: 'Trigger now' }) as HTMLButtonElement).disabled).toBe(false)
      expect(container.querySelector('[data-slot="cron-run-pending"]')).toBeNull()
    } finally {
      vi.clearAllTimers()
      vi.useRealTimers()
    }
  })

  it('removes pending feedback and unlocks the action when triggering fails', async () => {
    triggerCronJob.mockRejectedValue(new Error('scheduler unavailable'))
    const { container } = await renderCron()

    await screen.findByText('No runs yet')
    const triggerButton = screen.getByRole('button', { name: 'Trigger now' })

    await act(async () => triggerButton.click())

    await waitFor(() => expect(container.querySelector('[data-slot="cron-run-pending"]')).toBeNull())
    expect((triggerButton as HTMLButtonElement).disabled).toBe(false)
    expect(triggerButton.querySelector('.codicon-zap')).toBeTruthy()
  })
})
