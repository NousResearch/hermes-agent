import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { getCronJobOutputs } from '@/hermes'
import { TRANSLATIONS } from '@/i18n'

import { CronJobSidebarRuns } from './cron-jobs-section'

vi.mock('@/components/pane-shell/pane-visibility', () => ({
  usePaneVisible: () => true
}))

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getCronJobOutputs: vi.fn()
}))

describe('CronJobSidebarRuns', () => {
  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it('opens the durable output instead of a cron chat session', async () => {
    vi.mocked(getCronJobOutputs).mockResolvedValue([
      {
        byte_size: 72,
        created_at: 1_786_435_200,
        filename: '2026-08-11_09-00-00.md',
        id: '2026-08-11_09-00-00'
      }
    ])
    const onOpenRun = vi.fn()

    render(<CronJobSidebarRuns jobId="report-job" onOpenRun={onOpenRun} profile="worker_alpha" />)

    fireEvent.click(await screen.findByRole('button'))

    expect(getCronJobOutputs).toHaveBeenCalledWith('report-job', 5, 'worker_alpha')
    expect(onOpenRun).toHaveBeenCalledWith('report-job', '2026-08-11_09-00-00', 'worker_alpha')
  })

  it('does not describe a failed output request as an empty history', async () => {
    vi.mocked(getCronJobOutputs).mockRejectedValue(new Error('profile backend unavailable'))

    render(<CronJobSidebarRuns jobId="report-job" onOpenRun={vi.fn()} />)

    expect(await screen.findByText(TRANSLATIONS.en.cron.failedLoad)).toBeTruthy()
    expect(screen.queryByText(TRANSLATIONS.en.cron.noRuns)).toBeNull()
  })
})
