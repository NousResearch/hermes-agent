import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $cronFocusJobId, $cronJobs } from '@/store/cron'
import type { CronJob, SessionInfo } from '@/types/hermes'

import { CronView } from './index'

const api = vi.hoisted(() => ({
  createCronJob: vi.fn(),
  deleteCronJob: vi.fn(),
  getCronJobRuns: vi.fn(),
  getCronJobs: vi.fn(),
  pauseCronJob: vi.fn(),
  resumeCronJob: vi.fn(),
  setApiRequestProfile: vi.fn(),
  triggerCronJob: vi.fn(),
  updateCronJob: vi.fn()
}))

const runCache = vi.hoisted(() => ({
  getCachedCronRuns: vi.fn<(jobId: string, limit?: number) => null | SessionInfo[]>(),
  invalidateCronJobRuns: vi.fn<(jobId: string) => void>(),
  loadCronJobRuns: vi.fn<(jobId: string, limit?: number) => Promise<SessionInfo[]>>()
}))

vi.mock('@/hermes', () => api)

vi.mock('@/store/cron', async importOriginal => {
  const actual = await importOriginal<Record<string, unknown>>()

  return { ...actual, ...runCache }
})

function renderCronView() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  return render(
    <QueryClientProvider client={client}>
      <CronView onClose={vi.fn()} />
    </QueryClientProvider>
  )
}

const job: CronJob = {
  enabled: true,
  id: 'overlay-job',
  name: 'Overlay job',
  next_run_at: '2026-07-17T09:00:00Z',
  prompt: 'Run the report',
  state: 'enabled'
}

describe('CronView run history cache', () => {
  beforeEach(() => {
    for (const mock of Object.values(api)) {
      mock.mockReset()
    }

    runCache.getCachedCronRuns.mockReset()
    runCache.getCachedCronRuns.mockReturnValue([])
    runCache.invalidateCronJobRuns.mockReset()
    runCache.loadCronJobRuns.mockReset()
    runCache.loadCronJobRuns.mockResolvedValue([])
    api.getCronJobs.mockResolvedValue([job])
    api.triggerCronJob.mockResolvedValue(job)
    api.deleteCronJob.mockResolvedValue({ ok: true })
    $cronJobs.set([job])
    $cronFocusJobId.set(null)
  })

  afterEach(() => {
    cleanup()
    $cronJobs.set([])
    $cronFocusJobId.set(null)
  })

  it('reads and refreshes the default 20-row overlay cache shape', async () => {
    renderCronView()

    expect(runCache.getCachedCronRuns).toHaveBeenCalledWith('overlay-job')
    await waitFor(() => expect(runCache.loadCronJobRuns).toHaveBeenCalledWith('overlay-job'))
  })

  it('invalidates all run-history shapes after triggering a job', async () => {
    renderCronView()

    fireEvent.click(screen.getByRole('button', { name: 'Trigger now' }))

    await waitFor(() => expect(api.triggerCronJob).toHaveBeenCalledWith('overlay-job'))
    expect(runCache.invalidateCronJobRuns).toHaveBeenCalledWith('overlay-job')
  })

  it('invalidates all run-history shapes after deleting a job', async () => {
    renderCronView()

    fireEvent.pointerDown(screen.getByRole('button', { name: 'Actions' }), {
      button: 0,
      ctrlKey: false,
      pointerType: 'mouse'
    })
    fireEvent.click(await screen.findByRole('menuitem', { name: 'Delete' }))
    fireEvent.click(await screen.findByRole('button', { name: 'Delete' }))

    await waitFor(() => expect(api.deleteCronJob).toHaveBeenCalledWith('overlay-job'))
    expect(runCache.invalidateCronJobRuns).toHaveBeenCalledWith('overlay-job')
  })
})
