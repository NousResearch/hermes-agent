// @vitest-environment jsdom
import { QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as HermesApi from '@/hermes'
import type { CronJob, SessionInfo } from '@/hermes'
import { queryClient } from '@/lib/query-client'
import { $cronFocusJobId, $cronJobs, setCronFocusJobId, setCronJobs } from '@/store/cron'
import { notifyCronChanged, setChangeEventsAvailable } from '@/store/live-sync'

const getCronJobRuns = vi.fn()
const getCronJobs = vi.fn()
const triggerCronJob = vi.fn()

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof HermesApi>()),
  getCronJobRuns: (jobId: string) => getCronJobRuns(jobId),
  getCronJobs: (profile?: string) => getCronJobs(profile),
  triggerCronJob: (jobId: string) => triggerCronJob(jobId)
}))

vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: vi.fn()
}))

// No runs yet: the queued trigger row must carry the Run History section on
// its own, which is exactly the gap the issue reports (#70826).
const job: CronJob = {
  deliver: 'local',
  enabled: true,
  id: 'job-1',
  name: 'Status report',
  next_run_at: '2026-10-06T12:00:00Z',
  prompt: 'Prepare a status report',
  schedule: { expr: '*/5 * * * *', kind: 'cron' },
  schedule_display: 'Every 5 minutes',
  state: 'scheduled'
} as CronJob

const queuedRow = () => window.document.querySelector('[data-slot="cron-run-pending"]')

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

beforeAll(() => {
  HTMLElement.prototype.scrollIntoView ??= () => undefined
  vi.stubGlobal('CSS', { escape: (value: string) => value })
})

beforeEach(() => {
  setCronJobs([job])
  setCronFocusJobId(null)
  setChangeEventsAvailable(false)
  getCronJobs.mockResolvedValue([job])
  getCronJobRuns.mockResolvedValue([])
  triggerCronJob.mockResolvedValue(job)
})

afterEach(() => {
  vi.clearAllTimers()
  vi.useRealTimers()
  cleanup()
  queryClient.clear()
  setCronJobs([])
  setCronFocusJobId(null)
  setChangeEventsAvailable(false)
  vi.clearAllMocks()
})

describe('CronView queued trigger feedback (#70826)', () => {
  it('paints the queued run row immediately and settles it when the run appears', async () => {
    const { container } = await renderLoadedCron()
    const triggerButton = screen.getByRole('button', { name: 'Trigger now' })

    // The trigger request is still in flight — the queued row is pure
    // optimism from the click, not a backend response.
    let releaseTrigger!: (value: CronJob) => void
    triggerCronJob.mockReturnValue(new Promise<CronJob>(resolve => (releaseTrigger = resolve)))

    await act(async () => {
      triggerButton.click()
      await Promise.resolve()
    })

    expect(queuedRow()).toBeTruthy()
    expect(triggerButton.getAttribute('aria-busy')).toBe('true')
    expect(getCronJobRuns).toHaveBeenCalled()

    // The backend materializes the run the trigger produced.
    const observedAt = Date.now() / 1000
    getCronJobRuns.mockResolvedValue([
      {
        id: 'cron_job-1_20261006_120001',
        last_active: observedAt,
        started_at: observedAt,
        title: 'Status report run'
      } as SessionInfo
    ])
    await act(async () => {
      releaseTrigger(job)
    })

    await screen.findByText('Status report run')
    await waitFor(() => expect(queuedRow()).toBeNull())
    expect((triggerButton as HTMLButtonElement).disabled).toBe(false)
    expect(container.querySelector('.codicon-zap')).toBeTruthy()
  })

  it('reconciles the queued row immediately when cron.changed publishes the run', async () => {
    setChangeEventsAvailable(true)
    const { container } = await renderLoadedCron()
    const triggerButton = screen.getByRole('button', { name: 'Trigger now' })

    getCronJobRuns.mockClear()
    await act(async () => {
      triggerButton.click()
    })
    await waitFor(() => expect(getCronJobRuns).toHaveBeenCalled())

    // Between polls (the event-capable cadence is a slow backstop), the
    // broadcast must be enough to surface the run and settle the row.
    const callsBeforeChange = getCronJobRuns.mock.calls.length
    const observedAt = Date.now() / 1000
    getCronJobRuns.mockResolvedValue([
      {
        id: 'cron_job-1_20261006_120002',
        last_active: observedAt,
        started_at: observedAt,
        title: 'Status report run'
      } as SessionInfo
    ])
    await act(async () => {
      notifyCronChanged()
    })

    await waitFor(() => expect(getCronJobRuns.mock.calls.length).toBeGreaterThan(callsBeforeChange))
    await screen.findByText('Status report run')
    await waitFor(() => expect(queuedRow()).toBeNull())
    expect((triggerButton as HTMLButtonElement).disabled).toBe(false)
    expect(container.querySelector('[data-slot="cron-run-pending"]')).toBeNull()
  })

  it('polls fast while the run is queued and not before', async () => {
    await renderLoadedCron()

    getCronJobRuns.mockClear()
    vi.useFakeTimers()
    // Idle cadence: 8s legacy poll (change events unavailable in this test).
    await act(async () => {
      vi.advanceTimersByTimeAsync(3000)
    })
    expect(getCronJobRuns).not.toHaveBeenCalled()

    const triggerButton = screen.getByRole('button', { name: 'Trigger now' })

    await act(async () => {
      triggerButton.click()
      await Promise.resolve()
    })

    // Queued cadence: 1s pending poll fires well inside the idle interval.
    await act(async () => {
      vi.advanceTimersByTimeAsync(1500)
    })

    expect(getCronJobRuns.mock.calls.length).toBeGreaterThanOrEqual(1)
  })

  it('drops the queued row and unlocks the action at the bounded timeout', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true })
    await renderLoadedCron()

    const triggerButton = screen.getByRole('button', { name: 'Trigger now' })

    await act(async () => {
      triggerButton.click()
      await Promise.resolve()
    })

    expect(queuedRow()).toBeTruthy()

    // The trigger request itself is still pending (the real handler holds it
    // for the whole run), so only the 90s deadline can settle the row.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(90_000)
    })

    expect(queuedRow()).toBeNull()
    expect((triggerButton as HTMLButtonElement).disabled).toBe(false)
  })

  it('removes the queued row when the trigger request fails', async () => {
    await renderLoadedCron()
    const triggerButton = screen.getByRole('button', { name: 'Trigger now' })

    triggerCronJob.mockRejectedValue(new Error('scheduler unavailable'))

    await act(async () => {
      triggerButton.click()
    })

    await waitFor(() => expect(queuedRow()).toBeNull())
    expect((triggerButton as HTMLButtonElement).disabled).toBe(false)
  })

  it('does not let pause clear the queued run while the backend is still working', async () => {
    let releaseTrigger!: (value: CronJob) => void
    triggerCronJob.mockReturnValue(new Promise<CronJob>(resolve => (releaseTrigger = resolve)))
    const { container } = await renderLoadedCron()

    await act(async () => {
      screen.getByRole('button', { name: 'Trigger now' }).click()
      await Promise.resolve()
    })
    expect(queuedRow()).toBeTruthy()

    // The trigger is accepted but the run has not appeared; pausing the job
    // persists enabled=false but never cancels an in-flight execution
    // (cron/jobs.py pause path), so the queued row must survive it.
    const pauseButton = screen.getByRole('button', { name: 'Pause' })

    await act(async () => {
      pauseButton.click()
      await Promise.resolve()
    })

    expect(queuedRow()).toBeTruthy()
    expect(container.querySelector('[data-slot="cron-run-pending"]')).toBeTruthy()

    releaseTrigger(job)
  })
})

describe('CronView trigger duplicate-click guard (regression for f9d64b9 contract)', () => {
  it('coalesces a same-tick double click into one request', async () => {
    let releaseTrigger!: (value: CronJob) => void
    triggerCronJob.mockReturnValue(new Promise<CronJob>(resolve => (releaseTrigger = resolve)))
    await renderLoadedCron()

    const triggerButton = screen.getByRole('button', { name: 'Trigger now' })

    await act(async () => {
      triggerButton.click()
      triggerButton.click()
      await Promise.resolve()
    })

    expect(triggerCronJob).toHaveBeenCalledOnce()
    expect((triggerButton as HTMLButtonElement).disabled).toBe(true)

    await act(async () => {
      releaseTrigger(job)
    })
  })
})

// Keep the atom import referenced even when no test reads it directly.
void $cronJobs
void $cronFocusJobId
