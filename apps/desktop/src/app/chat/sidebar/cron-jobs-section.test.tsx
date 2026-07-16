import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { SessionInfo } from '@/types/hermes'

import { SidebarCronJobsSection } from './cron-jobs-section'

const getCachedCronRuns = vi.fn<(jobId: string, limit?: number) => null | SessionInfo[]>()
const loadCronJobRuns = vi.fn<(jobId: string, limit?: number) => Promise<SessionInfo[]>>()

vi.mock('@/store/cron', () => ({
  getCachedCronRuns: (jobId: string, limit?: number) => getCachedCronRuns(jobId, limit),
  loadCronJobRuns: (jobId: string, limit?: number) => loadCronJobRuns(jobId, limit)
}))

const job = {
  enabled: true,
  id: 'sidebar-job',
  name: 'Sidebar job',
  next_run_at: '2026-07-17T09:00:00Z',
  state: 'enabled'
}

describe('SidebarCronJobsSection run history', () => {
  beforeEach(() => {
    getCachedCronRuns.mockReset()
    getCachedCronRuns.mockReturnValue(null)
    loadCronJobRuns.mockReset()
    loadCronJobRuns.mockResolvedValue([])
  })

  afterEach(cleanup)

  it('reads and refreshes the five-row sidebar cache shape', async () => {
    render(
      <SidebarCronJobsSection
        jobs={[job]}
        label="Cron jobs"
        onManageJob={vi.fn()}
        onOpenRun={vi.fn()}
        onToggle={vi.fn()}
        onTriggerJob={vi.fn()}
        open
      />
    )

    fireEvent.click(screen.getByRole('button', { name: 'Show runs' }))

    expect(getCachedCronRuns).toHaveBeenCalledWith('sidebar-job', 5)
    await waitFor(() => expect(loadCronJobRuns).toHaveBeenCalledWith('sidebar-job', 5))
  })
})
